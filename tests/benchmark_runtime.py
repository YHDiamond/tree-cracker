#!/usr/bin/env python3
"""Compare data-ready to complete 48-bit candidate output on one CUDA GPU.

Source checkout/download and GPU provisioning are outside the measurement.
Compiled observations pay preparation + nvcc + startup + exhaustive search on
every trial. Runtime observations pay preparation + startup + exhaustive search;
their reusable build is recorded separately and added to cold-start totals.
Both paths receive identical observations, never a seed or a search hint.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ['-O3', '-arch=sm_75', '-Xcompiler=-mcmodel=large',
         '-Xlinker=--no-relax', '--cudart=shared']


def hardware():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total',
                                    '--format=csv,noheader'], text=True).strip()


def compiled_header(observations):
    rows = []
    states = {'0': 'LeafWasNotPlaced', '1': 'LeafWasPlaced', '?': 'Unknown'}
    for line in observations.splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        version, biome, kind, x, z, lower, upper, leaves = fields
        heights = '' if lower == upper == '0' else lower + ', ' + upper
        corners = ', '.join('LeafState::' + states[c] for c in leaves)
        rows.append(f'{{Version::v{version.replace(".", "_")}, TreeType::{kind}, '
                    f'Coordinate({x}, {z}), Biome::{biome}, PossibleHeightsRange({heights}), '
                    f'std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({{{corners}}})}}')
    return '__device__ constexpr InputData INPUT_DATA[] = {\n' + ',\n'.join(rows) + '\n};'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', default='a69f236')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--expected-hardware', help='Require this exact UUID/name/driver/memory string')
    args = ap.parse_args()
    assert args.repeats > 0
    output = args.output.resolve()
    output.mkdir(exist_ok=False, parents=True)
    gpu = hardware()
    if args.expected_hardware:
        assert gpu == args.expected_hardware, (gpu, args.expected_hardware)
    revision = subprocess.check_output(['git', 'rev-parse', args.baseline], cwd=ROOT, text=True).strip()
    baseline = output / 'compiled'
    baseline.mkdir()
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git', 'archive', revision], cwd=ROOT))) as archive:
        archive.extractall(baseline, filter='data')
    settings = (baseline / 'Settings (MODIFY THIS).cuh').read_text()
    # Both paths flush candidate stdout so the first-output time is observable.
    main_file = baseline / 'main.cu'
    main_text = main_file.read_text()
    line = next(line for line in main_text.splitlines() if 'if (!SILENT_MODE) std::printf' in line and 'ACTUAL_TYPES_TO_OUTPUT == OutputType::Structure_Seeds' in line)
    main_file.write_text(main_text.replace(line, line + '\n\t\t\t\t\tstd::fflush(stdout);', 1))
    runtime_binary = output / 'runtime-main'
    start = time.monotonic()
    with (output / 'runtime-build.log').open('w') as log:
        subprocess.run(['nvcc', 'main.cu', '-o', str(runtime_binary)] + FLAGS, cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    build_seconds = time.monotonic() - start
    cases = {
        'fresh-four-tree': (ROOT / 'Test Data/fresh-16-1-20261009.txt').read_text(),
        'documented-five-tree': (ROOT / 'Test Data/documented-16-1.txt').read_text(),
    }
    report = {'hardware': gpu, 'compiler': subprocess.check_output(['nvcc', '--version'], text=True).strip(),
              'runtime_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'baseline_commit': revision, 'flags': FLAGS, 'runtime_build_seconds': build_seconds,
              'metric': 'observations_available_to_complete_exhaustive_candidate_output',
              'exhaustive': True, 'domain_states': 1 << 44, 'seed_supplied_to_solver': False,
              'baseline_stdout_flush_only_change': True, 'repeats': args.repeats, 'results': []}
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + env.get('LD_LIBRARY_PATH', '')
    print('BUILD runtime', build_seconds, flush=True)
    references = {}
    for case, observations in cases.items():
        for repeat in range(args.repeats):
            order = ['compiled', 'runtime'] if repeat % 2 == 0 else ['runtime', 'compiled']
            for variant in order:
                assert hardware() == gpu
                work = output / f'{case}-{repeat}-{variant}'
                work.mkdir()
                start = time.monotonic()  # All tree observations are now available.
                if variant == 'compiled':
                    header = compiled_header(observations)
                    a = settings.index('__device__ constexpr InputData INPUT_DATA[] = {')
                    b = settings.index('\n};', a) + 3
                    (baseline / 'Settings (MODIFY THIS).cuh').write_text(settings[:a] + header + settings[b:])
                    compile_start = time.monotonic()
                    with (work / 'build.log').open('w') as log:
                        subprocess.run(['nvcc', 'main.cu', '-o', str(work / 'main')] + FLAGS,
                                       cwd=baseline, stdout=log, stderr=subprocess.STDOUT, check=True)
                    compile_seconds = time.monotonic() - compile_start
                    command = [str(work / 'main')]
                else:
                    (work / 'observations.txt').write_text(observations)
                    compile_seconds = 0.
                    command = [str(runtime_binary), str(work / 'observations.txt')]
                solve_start = time.monotonic()
                first_output_seconds = None
                with (work / 'run.log').open('w') as log:
                    proc = subprocess.Popen(command, cwd=work, env=env, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    for line in proc.stdout:
                        if re.fullmatch(r'\d+\s*', line) and first_output_seconds is None:
                            first_output_seconds = time.monotonic() - start
                        log.write(line)
                    exit_code = proc.wait()
                finish = time.monotonic()
                text = (work / 'run.log').read_text()
                candidates = sorted(set(map(int, (work / 'output.txt').read_text().splitlines())))
                valid = exit_code == 0 and 'ignoring last' not in text
                assert valid and candidates, (case, variant, exit_code)
                references.setdefault(case, candidates)
                assert candidates == references[case], (case, variant, candidates, references[case])
                row = {'case': case, 'variant': variant, 'repeat': repeat,
                       'end_to_end_seconds': finish - start, 'solver_seconds': finish - solve_start,
                       'compile_seconds': compile_seconds, 'first_candidate_seconds': first_output_seconds,
                       'cold_end_to_end_seconds': finish - start + (build_seconds if variant == 'runtime' else 0.),
                       'input_sha256': hashlib.sha256(observations.encode()).hexdigest(),
                       'candidate_set': candidates, 'exit_code': exit_code, 'overflow': False, 'same_output': True}
                report['results'].append(row)
                (output / 'report.json').write_text(json.dumps(report, indent=2))
                print('TRIAL ' + json.dumps(row), flush=True)
    report['summary'] = {}
    for case in cases:
        summary = {}
        for variant in ['compiled', 'runtime']:
            rows = [r for r in report['results'] if r['case'] == case and r['variant'] == variant]
            summary[variant] = {key: statistics.median(r[key] for r in rows) for key in
                                ['end_to_end_seconds', 'solver_seconds', 'compile_seconds',
                                 'first_candidate_seconds', 'cold_end_to_end_seconds']}
        summary['end_to_end_speedup'] = summary['compiled']['end_to_end_seconds'] / summary['runtime']['end_to_end_seconds']
        report['summary'][case] = summary
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print('RUNTIME_COMPARISON ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
