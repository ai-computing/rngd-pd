"""Standalone research figures from final n5 raw-linked E6 evidence only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
summary = json.loads((root / 'e6-summary.json').read_bytes())
assert summary['complete'] and len(summary['groups']) == 6
assert all(g['headline_eligible'] and g['n_valid'] == 5 for g in summary['groups'])
collection = summary['collection']
assert summary['collection_sha256'] == hashlib.sha256((root / 'e6-collection.json').read_bytes()).hexdigest()
for sample in collection['samples']:
    assert hashlib.sha256((root / sample['raw_path']).read_bytes()).hexdigest() == sample['raw_sha256']
groups = {(g['group']['cache_state'], g['group']['arm']): g for g in summary['groups']}
arms = ['full', 'delta', 'resident']
labels = ['L1\nFull + restart', 'L1+L2\nDelta + restart', 'L1+L2+L3\nResident']
cache_states = ['empty', 'prepopulated']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
out = root / 'figures'
out.mkdir(exist_ok=True)


def export(fig, name):
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(out / (name + '.' + suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
for ax, cache in zip(axes, cache_states):
    x = np.arange(3)
    for metric, color, marker, offset, label in [
        ('pod_creation_to_first_token_s', '#59636d', 's', -.08, 'Pod creation to first token'),
        ('reconfiguration_to_first_token_s', '#167c80', 'o', .08, 'Trigger to first token')]:
        stats = [groups[(cache, arm)]['metrics'][metric] for arm in arms]
        medians = np.array([m['median'] for m in stats])
        errors = np.array([[m['median'] - m['q1'] for m in stats],
                           [m['q3'] - m['median'] for m in stats]])
        ax.errorbar(x + offset, medians, yerr=errors, fmt=marker, color=color,
                    capsize=4, markersize=5, label=label)
        for xi, value in zip(x + offset, medians):
            ax.annotate(f'{value:.2f}', (xi, value), xytext=(5, 0),
                        textcoords='offset points', fontsize=7, va='center')
    ax.set_yscale('log')
    ax.set_xticks(x, labels)
    ax.set_xlim(-.35, 2.55)
    ax.set_title('Node chunk cache: ' + cache)
    ax.grid(axis='y', alpha=.2, which='both')
axes[0].set_ylabel('Seconds (log scale)')
axes[0].legend(loc='lower left', frameon=False, fontsize=8)
fig.suptitle('RNGD B4/S256 → B4/S512: median and IQR, n=5 per condition', fontsize=10)
fig.tight_layout()
export(fig, 'e6-ttft')

# A stacked bar must add to its displayed total. Use the actual middle-total
# trial, rather than summing three unrelated marginal component medians.
selected = []
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
phase_keys = ['artifact_delivery_s', 'delivery_to_generation_start_s', 'target_prompt_to_first_token_s']
phase_names = ['Artifact delivery', 'Delivery to generation', 'Native prompt TTFT']
colors = ['#4477aa', '#eeaa44', '#44aa99']
for ax, cache in zip(axes, cache_states):
    rows = []
    for arm in arms:
        trials = sorted([s for s in collection['samples'] if s['cache_state'] == cache and s['arm'] == arm],
                        key=lambda s: s['metrics']['reconfiguration_to_first_token_s'])
        row = trials[2]
        assert row['metrics']['reconfiguration_to_first_token_s'] == groups[(cache, arm)]['metrics']['reconfiguration_to_first_token_s']['median']
        rows.append(row)
        selected.append(dict(cache_state=cache, arm=arm, trial_id=row['trial_id'],
            raw_path=row['raw_path'], raw_sha256=row['raw_sha256'],
            phases={k: row['metrics'][k] for k in phase_keys}))
    bottom = np.zeros(3)
    for key, label, color in zip(phase_keys, phase_names, colors):
        values = [row['metrics'][key] for row in rows]
        ax.bar(range(3), values, bottom=bottom, color=color, width=.55, label=label)
        bottom += values
    for x, total in enumerate(bottom):
        assert abs(total - rows[x]['metrics']['reconfiguration_to_first_token_s']) < 1e-6
        ax.text(x, total + .7, f'{total:.2f}', ha='center', fontsize=8)
    ax.set_xticks(range(3), labels)
    ax.set_title('Node chunk cache: ' + cache)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Trigger to first native token (seconds)')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False, fontsize=8)
fig.suptitle('Stage breakdown of the trial with median total time', fontsize=10)
fig.tight_layout(rect=(0, .08, 1, 1))
export(fig, 'e6-trigger-phases')
(out / 'provenance.json').write_text(json.dumps(dict(
    plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    summary_sha256=hashlib.sha256((root / 'e6-summary.json').read_bytes()).hexdigest(),
    matplotlib=matplotlib.__version__, n=5,
    phase_selection='actual trial with median total, not sum of marginal medians',
    selected_trials=selected,
    scope_notes=collection['conditions']), indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(dict(figures=2, formats=['png', 'svg', 'pdf'], selected_trials=len(selected))))
