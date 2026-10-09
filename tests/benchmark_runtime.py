#!/usr/bin/env python3
"""Compare reusable CUDA solvers from observations available to correct seed output.

The expected seed is validation-only: it is consulted after the full search,
never passed to a solver, used to restrict its domain, or used to stop a run.
Input writing, process/CUDA startup, parsing, search and output are timed.
Source checkout, GPU allocation and each reusable build are outside that time.
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
CASES = {
    'fresh-four-tree': ('fresh-16-1-20261009.txt', 128110318218222,
                        [128110318218222]),
    'documented-five-tree': ('documented-16-1.txt', 157527116063087,
                             [89593286004526, 115468527086863,
                              157527116063087, 272740909957168]),
}


def hardware():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total',
                                    '--format=csv,noheader'], text=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', action='append', required=True, help='name=revision; first is baseline')
    ap.add_argument('--source-repo', type=Path, default=ROOT)
    ap.add_argument('--reuse-binary', action='append', default=[], help='name=already built binary path')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--expected-hardware', required=True, help='Exact UUID/name/driver/memory string')
    args = ap.parse_args()
    assert args.repeats > 0 and len(args.variant) >= 2
    output = args.output.resolve()
    output.mkdir(exist_ok=False, parents=True)
    gpu = hardware()
    assert gpu == args.expected_hardware, (gpu, args.expected_hardware)
    reused = dict(item.split('=', 1) for item in args.reuse_binary)
    variants = {}
    for item in args.variant:
        name, ref = item.split('=', 1)
        assert re.fullmatch(r'[a-zA-Z0-9_-]+', name) and name not in variants
        revision = subprocess.check_output(['git', 'rev-parse', ref], cwd=args.source_repo, text=True).strip()
        source = output / name
        source.mkdir()
        with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git', 'archive', revision], cwd=args.source_repo))) as archive:
            archive.extractall(source, filter='data')
        binary = Path(reused[name]).resolve() if name in reused else source / 'main'
        build_seconds = None
        if name not in reused:
            start = time.monotonic()
            with (source / 'build.log').open('w') as log:
                subprocess.run(['nvcc', 'main.cu', '-o', str(binary)] + FLAGS,
                               cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
            build_seconds = time.monotonic() - start
        variants[name] = {'revision': revision, 'binary': str(binary), 'build_seconds': build_seconds,
                          'build_reused': name in reused,
                          'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
        print('BUILD ' + json.dumps({name: variants[name]}), flush=True)
    report = {'hardware': gpu, 'compiler': subprocess.check_output(['nvcc', '--version'], text=True).strip(),
              'variants': variants, 'flags': FLAGS,
              'metric': 'observations_available_to_correct_seed_emitted',
              'validation_only_truth_checked_after_exhaustive_run': True,
              'exhaustive': True, 'domain_states': 1 << 44, 'seed_supplied_to_solver': False,
              'timing_boundary': {'start': 'complete tree observations available in active GPU environment',
                                  'includes': ['input writing', 'process and CUDA startup', 'parsing and chunk construction',
                                               'search until correct seed printed'],
                                  'excludes': ['observation collection', 'GPU provisioning', 'source download', 'reusable build']},
              'repeats': args.repeats, 'results': []}
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + env.get('LD_LIBRARY_PATH', '')
    for case, (filename, truth, expected) in CASES.items():
        observations = (ROOT / 'Test Data' / filename).read_text()
        for repeat in range(args.repeats):
            order = list(variants) if repeat % 2 == 0 else list(reversed(variants))
            for name in order:
                assert hardware() == gpu
                work = output / f'{case}-{repeat}-{name}'
                work.mkdir()
                start = time.monotonic()  # Observation contents are now available.
                (work / 'observations.txt').write_text(observations)
                emissions = []
                with (work / 'run.log').open('w') as log:
                    proc = subprocess.Popen([variants[name]['binary'], str(work / 'observations.txt')],
                                            cwd=work, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    for line in proc.stdout:
                        if re.fullmatch(r'\d+\s*', line):
                            emissions.append((int(line), time.monotonic() - start))
                        log.write(line)
                    exit_code = proc.wait()
                finish = time.monotonic()
                log_text = (work / 'run.log').read_text()
                candidates = sorted(set(map(int, (work / 'output.txt').read_text().splitlines())))
                assert exit_code == 0 and 'ignoring last' not in log_text and candidates == expected, (case, name, exit_code, candidates)
                correct_seconds = min(seconds for seed, seconds in emissions if seed == truth)
                row = {'case': case, 'variant': name, 'repeat': repeat,
                       'correct_seed_seconds': correct_seconds, 'complete_output_seconds': finish - start,
                       'first_candidate_seconds': emissions[0][1],
                       'unique_seed': len(candidates) == 1,
                       'input_sha256': hashlib.sha256(observations.encode()).hexdigest(),
                       'candidate_set': candidates, 'candidate_emissions': emissions,
                       'exit_code': exit_code, 'overflow': False, 'same_output': True}
                report['results'].append(row)
                (output / 'report.json').write_text(json.dumps(report, indent=2))
                print('TRIAL ' + json.dumps(row), flush=True)
    report['summary'] = {}
    for case in CASES:
        report['summary'][case] = {}
        for name in variants:
            rows = [r for r in report['results'] if r['case'] == case and r['variant'] == name]
            report['summary'][case][name] = {key: statistics.median(r[key] for r in rows)
                for key in ['correct_seed_seconds', 'complete_output_seconds', 'first_candidate_seconds']}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print('RUNTIME_SPEED_REPORT ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
