"""Run numerical analyses without using saved results as computational inputs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Prepared model inputs')
    parser.add_argument('--biochar-source', type=Path)
    parser.add_argument('--activated-carbon-source', type=Path)
    parser.add_argument('--graph-data-dir', type=Path, required=True,
                        help='Directory containing the three documented Zhao input files')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    prepare = args.biochar_source is not None or args.activated_carbon_source is not None
    if args.data_dir is not None and prepare:
        parser.error('Choose prepared inputs or both source repositories, not both')
    if args.data_dir is None and not (args.biochar_source and args.activated_carbon_source):
        parser.error('Provide --data-dir or both source repository directories')
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output.resolve()
    if output.exists():
        parser.error(f'Output directory must not already exist: {output}')
    graph = args.graph_data_dir.resolve()
    graph_names = ('zhao_marker_pixels.csv', 'zhao_calibration.json', 'zhao_frozen_queries.csv')
    for name in graph_names:
        if not (graph / name).is_file():
            parser.error(f'Missing graphical input: {graph / name}')
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    cache = output / '.cache'
    cache.mkdir(mode=0o700)
    env = os.environ.copy()
    for name in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'MPLCONFIGDIR'):
        env[name] = str(cache)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    source = Path(__file__).resolve().parent
    steps = []
    started = time.monotonic()

    def run(script, arguments):
        command = [sys.executable, str(source / script), *map(str, arguments)]
        begin = time.monotonic()
        with (output / f'{Path(script).stem}.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        steps.append(dict(script=script, exit_code=result.returncode,
                          elapsed_seconds=time.monotonic() - begin))
        (output / 'execution.json').write_text(json.dumps(steps, indent=2) + '\n')
        if result.returncode:
            raise SystemExit(f'{script} failed; inspect its log in {output}')

    if prepare:
        data = output / 'prepared_data'
        run('prepare_inputs.py', ['--biochar-source', args.biochar_source.resolve(),
                                 '--activated-carbon-source', args.activated_carbon_source.resolve(),
                                 '--output', data])
    else:
        data = args.data_dir.resolve()
    run('reproduce.py', ['--data-dir', data, '--output', output / 'primary'])
    run('review_audit.py', ['--data-dir', data, '--output', output / 'sensitivity',
                           '--workers', args.workers])
    run('reproduce_supplement.py', ['--data-dir', graph, '--output', output / 'supplement'])
    inventory = []
    for path in sorted(output.rglob('*')):
        if path.is_file() and '.cache' not in path.relative_to(output).parts:
            inventory.append(dict(path=str(path.relative_to(output)), bytes=path.stat().st_size,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'pandas', 'scipy', 'scikit-learn', 'h5py', 'openpyxl')}
    report = dict(completed=True, python=platform.python_version(), versions=versions,
                  steps=steps, elapsed_seconds=time.monotonic() - started, outputs=inventory,
                  saved_results_used_as_inputs=False,
                  scope='Primary numerical results, sensitivity analyses, and supplementary controls; '
                        'not historical Kernel SHAP regression, source TensorFlow replay, or figure layout.')
    (output / 'run_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Numerical calculations completed. Report: {output / "run_report.json"}')


if __name__ == '__main__':
    main()
