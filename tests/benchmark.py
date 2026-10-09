#!/usr/bin/env python3
"""Compile and compare five fixed tree workloads on one CUDA device.

Default runs are exhaustive. --profile-batches is an explicitly incomplete
performance diagnostic; its results must never be called recovered seeds.
--positive-window is a validation-only regression test using known seeds to
derive nonempty bounded windows; it is never an inference or recovery run.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ['-O3', '-arch=sm_75', '-Xcompiler=-mcmodel=large',
         '-Xlinker=--no-relax', '--cudart=shared']


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def prepare(revision, fixture, destination, workers, capacity, profile_batches, positive_window=False):
    destination.mkdir(parents=True, exist_ok=False)
    archive = subprocess.check_output(['git', 'archive', revision], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(destination, filter='data')
    # All candidates use the same correctness repair, independently of speed edits.
    p = destination / 'src/Settings and Input Data Processing.cuh'
    text = p.read_text()
    for declaration in ['uint64_t filterStorageB[',
                        'DoubleStorage filterDoubleStorageA[',
                        'DoubleStorage filterDoubleStorageB[']:
        text = text.replace('__device__ ' + declaration, '__managed__ ' + declaration)
    p.write_text(text)
    trees = fixture['input']['trees']
    assert fixture['input']['biome'] == 'Forest'
    assert fixture['input']['minecraft_version'] == '1.16.1'
    assert all(set(tree) == {'x', 'z', 'type'} for tree in trees)
    rows = [f'    {{Version::v1_16_1, TreeType::{t["type"]}, '
            f'Coordinate({t["x"]}, {t["z"]}), Biome::Forest}}' for t in trees]
    p = destination / 'Settings (MODIFY THIS).cuh'
    text = p.read_text()
    a = text.index('__device__ constexpr InputData INPUT_DATA[] = {')
    b = text.index('\n};', a) + 3
    text = text[:a] + '__device__ constexpr InputData INPUT_DATA[] = {\n' + ',\n'.join(rows) + '\n};' + text[b:]
    text = text.replace('NUMBER_OF_WORKERS = 4294967296;', f'NUMBER_OF_WORKERS = {workers};')
    text = text.replace('MAX_NUMBER_OF_RESULTS_PER_RUN = AUTO;', f'MAX_NUMBER_OF_RESULTS_PER_RUN = {capacity};')
    text = text.replace('PRINT_TIMESTAMPS_FREQUENCY = 256;', 'PRINT_TIMESTAMPS_FREQUENCY = 1;')
    text = text.replace('"output.txt"', '"results.txt"')
    p.write_text(text)
    # The original primary kernel launches one extra block. Clip that block
    # for a controlled diagnostic window, so every variant tests the same states.
    p = destination / 'src/Filters.cuh'
    text = p.read_text()
    old = '__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter1(const uint64_t start) {\n\tuint32_t index = blockIdx.x * blockDim.x + threadIdx.x;'
    new = '__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter1(const uint64_t start) {\n\tuint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n\tif (index >= NUMBER_OF_WORKERS) FILTER_RETURN;'
    text = text.replace(old, new)
    begin = text.index('__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void treechunkFilter()')
    end = text.index('/* Reverses population seeds', begin)
    section = text[begin:end]
    if 'uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;' in section:
        fixed = subprocess.check_output(['git', 'show', '87d3675:src/Filters.cuh'], cwd=ROOT, text=True)
        a = fixed.index('__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void treechunkFilter()')
        b = fixed.index('/* Reverses population seeds', a)
        text = text[:begin] + fixed[a:b] + text[end:]
    p.write_text(text)
    if profile_batches:
        p = destination / 'main.cu'
        text = p.read_text()
        text = text.replace('partialRun <= ACTUAL_NUMBER_OF_PARTIAL_RUNS', 'partialRun <= ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM')
        marker = '\n\t\tif (!SILENT_MODE) {\n\t\t\tstd::fprintf(stderr, "Beginning partial run'
        assert marker in text
        text = text.replace(marker, f'\n\t\trunEndSeed = constexprMin(runEndSeed, runStartSeed + UINT64_C({workers * profile_batches}));' + marker, 1)
        p.write_text(text)
    if positive_window:
        # Use this revision's actual production launch, including any kernel ABI
        # changes. Neither the launch predicates nor production search are altered.
        main_source = (destination / 'main.cu').read_text()
        begin = main_source.index('// Filter 1 (states that can exactly generate')
        begin = main_source.index('#if CUDA_IS_PRESENT', begin) + len('#if CUDA_IS_PRESENT')
        end = main_source.index('#else', begin)
        launch = main_source[begin:end].strip()
        assert 'filter1<<<' in launch
        test_source = (ROOT / 'tests/positive_windows.cu').read_text()
        assert test_source.count('// FILTER1_LAUNCH_FROM_MAIN') == 1
        test_source = test_source.replace('// FILTER1_LAUNCH_FROM_MAIN', launch)
        begin = main_source.index('// Treechunk filter (States able to generate')
        begin = main_source.index('#if CUDA_IS_PRESENT', begin) + len('#if CUDA_IS_PRESENT')
        end = main_source.index('#else', begin)
        launch = main_source[begin:end].strip()
        assert 'treechunkFilter<<<' in launch
        assert test_source.count('// TREECHUNK_LAUNCH_FROM_MAIN') == 1
        test_source = test_source.replace('// TREECHUNK_LAUNCH_FROM_MAIN', launch)
        (destination / 'tests').mkdir(exist_ok=True)
        (destination / 'tests/positive_windows.cu').write_text(test_source)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', action='append', required=True, help='name=commit; first is baseline')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, help='batch/window size; default 2^28, or 2^20 for positive regression')
    ap.add_argument('--capacity', type=int, default=1 << 24)
    ap.add_argument('--profile-batches', type=int, default=0)
    ap.add_argument('--positive-window', action='store_true', help='validation-only known-positive bounded-window regression; never inference')
    ap.add_argument('--reference-report', type=Path, help='compare positive tests with an earlier report from this same GPU and configuration')
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--case', action='append', help='optional fixture ID filter')
    args = ap.parse_args()
    if args.positive_window and args.profile_batches:
        ap.error('--positive-window and --profile-batches are distinct modes; select one')
    if args.workers is None:
        args.workers = 1 << (20 if args.positive_window else 28)
    if args.positive_window and (args.workers > 1 << 24 or args.workers & (args.workers - 1)):
        ap.error('positive-window regression requires a power-of-two --workers <= 2^24')
    args.output = args.output.resolve()
    assert args.workers > 0 and args.capacity > 0 and args.profile_batches >= 0 and args.repeats > 0
    fixtures = json.loads((ROOT / 'Test Data/positions_types_fixture_candidates.json').read_text())['fixtures']
    assert len(fixtures) == 5 and len({f['validation_only']['world_seed'] for f in fixtures}) == 5
    if args.case:
        fixtures = [f for f in fixtures if f['id'] in args.case]
        assert len(fixtures) == len(args.case)
    args.output.mkdir(parents=True, exist_ok=False)
    gpu = command(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader'])
    report = {'hardware': gpu, 'compiler': command(['nvcc', '--version']),
              'exhaustive': not bool(args.profile_batches or args.positive_window),
              'mode': 'positive_window_regression' if args.positive_window else 'first_window_diagnostic' if args.profile_batches else 'exhaustive_inference',
              'validation_only': args.positive_window,
              'workers': args.workers, 'capacity': args.capacity, 'flags': FLAGS,
              'profile_batches': args.profile_batches, 'repeats': args.repeats,
              'controlled_worker_bounds': True, 'corrected_64bit_treechunk_indices': True, 'results': []}
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + env.get('LD_LIBRARY_PATH', '')
    references = {}
    reference_windows = {}
    reference_inputs = {}
    if args.reference_report:
        assert args.positive_window, 'Saved candidate sets are supported for positive regressions'
        previous = json.loads(args.reference_report.read_text())
        for key in ['hardware', 'compiler', 'mode', 'workers', 'capacity', 'flags']:
            assert previous[key] == report[key], (key, previous[key], report[key])
        for row in previous['results']:
            if row['case'] in references:
                continue
            assert row['valid'] and row['same_output'] and row['expected_positive_retained']
            references[row['case']] = (row['candidate_set'], True)
            reference_windows[row['case']] = row['tested_state_window']
            reference_inputs[row['case']] = row['input_sha256']
        assert all(f['id'] in references for f in fixtures)
        report['reference_report'] = str(args.reference_report)
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)
    for variant in args.variant:
        name, revision = variant.split('=', 1)
        sha = command(['git', 'rev-parse', revision], cwd=ROOT)
        for fixture in fixtures:
            assert command(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader']) == gpu
            target = args.output / name / fixture['id']
            prepare(sha, fixture, target, args.workers, args.capacity, args.profile_batches, args.positive_window)
            start = time.monotonic()
            with (target / 'build.log').open('w') as log:
                source = 'tests/positive_windows.cu' if args.positive_window else 'main.cu'
                subprocess.run(['nvcc', source, '-o', 'main'] + FLAGS, cwd=target, stdout=log, stderr=subprocess.STDOUT, check=True)
            build_seconds = time.monotonic() - start
            print(f'RUN {name} {fixture["id"]} build={build_seconds:.3f}s', flush=True)
            for repeat in range(args.repeats):
                start = time.monotonic()
                with (target / f'run-{repeat}.log').open('w') as log:
                    run_args = [str(target / 'main')]
                    if args.positive_window:
                        run_args.append(fixture['validation_only']['structure_seed'])
                    proc = subprocess.Popen(run_args, cwd=target, env=env,
                                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    exit_code = proc.wait()
                seconds = time.monotonic() - start
                log_text = (target / f'run-{repeat}.log').read_text()
                result_file = target / 'results.txt'
                candidates = sorted(set(map(int, result_file.read_text().splitlines()))) if result_file.exists() else []
                valid = exit_code == 0 and 'ignoring last' not in log_text
                digest = hashlib.sha256(json.dumps(candidates, separators=(',', ':')).encode()).hexdigest()
                row = {'variant': name, 'commit': sha, 'case': fixture['id'], 'seconds': seconds,
                       'build_seconds': build_seconds, 'repeat': repeat, 'exit_code': exit_code, 'valid': valid,
                       'candidate_count': len(candidates), 'candidate_sha256': digest,
                       'expected_recovered': int(fixture['validation_only']['structure_seed']) in candidates,
                       'input_sha256': hashlib.sha256(json.dumps(fixture['input'], sort_keys=True).encode()).hexdigest()}
                if reference_inputs:
                    assert row['input_sha256'] == reference_inputs[fixture['id']]
                if args.positive_window:
                    timing = re.search(r'passed=true exhaustive=false seconds=([0-9.]+)', log_text)
                    row['pipeline_seconds'] = float(timing.group(1)) if timing else None
                    match = re.search(r'window=\[(\d+),(\d+)\)', log_text)
                    row['tested_state_window'] = list(map(int, match.groups())) if match else None
                    row['validation_only'] = True
                    row['expected_positive_retained'] = int(fixture['validation_only']['structure_seed']) in candidates
                    row['candidate_set'] = candidates
                    if fixture['id'] not in reference_windows:
                        reference_windows[fixture['id']] = row['tested_state_window']
                    row['same_window'] = row['tested_state_window'] == reference_windows[fixture['id']]
                    valid = valid and bool(match) and row['same_window'] and 'UNIT_REGRESSION passed=true' in log_text
                    row['valid'] = valid
                if fixture['id'] not in references:
                    references[fixture['id']] = (candidates, valid)
                    row['same_output'] = True
                else:
                    previous, previous_valid = references[fixture['id']]
                    row['same_output'] = valid and previous_valid and candidates == previous
                report['results'].append(row)
                (args.output / 'report.json').write_text(json.dumps(report, indent=2))
                if result_file.exists():
                    result_file.rename(target / f'results-{repeat}.txt')
                print('RESULT ' + json.dumps(row), flush=True)
                assert valid, row
                assert row['same_output'], row
                if report['exhaustive']:
                    assert row['expected_recovered'], row
                if args.positive_window:
                    assert row['expected_positive_retained'], row



if __name__ == '__main__':
    main()
