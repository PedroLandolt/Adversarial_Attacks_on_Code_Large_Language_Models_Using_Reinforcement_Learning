"""Run plot.py against all stored_results recursively, output to Pedro_MSc/plots/."""
import sys, json
from pathlib import Path

sys.path.insert(0, 'V3')

stored = Path('stored_results')
run_dirs = []
for p in sorted(stored.rglob('run_summary.json')):
    d = p.parent
    if (d / 'run_config.json').exists() and (d / 'attempts.jsonl').exists():
        run_dirs.append(d)
print(f'Found {len(run_dirs)} run dirs in stored_results')

loaded = []
for d in run_dirs:
    try:
        rc = json.loads((d / 'run_config.json').read_text(encoding='utf-8'))
        rs = json.loads((d / 'run_summary.json').read_text(encoding='utf-8'))
        atts = [
            json.loads(l) for l in
            (d / 'attempts.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()
        ]
        loaded.append({'run_path': str(d), 'run_config': rc, 'run_summary': rs, 'attempts': atts})
    except Exception as e:
        print(f'Skip {d}: {e}')

print(f'Loaded {len(loaded)} runs')

import utils.results_persistence as rp
_orig = rp.load_persisted_runs
rp.load_persisted_runs = lambda *a, **k: loaded

from utils.results_aggregation import aggregate_persisted_runs
aggregation = aggregate_persisted_runs(results_dir='stored_results')

rp.load_persisted_runs = _orig
nr = len(aggregation['runs'])
ng = len(aggregation['grouped_summary'])
print(f'Aggregated: {nr} rows, {ng} groups')

from plot import generate_plots
out = Path('../Pedro_MSc/plots')
out.mkdir(parents=True, exist_ok=True)
manifest = generate_plots(aggregation, str(out))
print(f'Generated {manifest["plot_count"]} plots -> {out}')
for p in manifest['plots']:
    print(f'  -> {Path(p).name}')
