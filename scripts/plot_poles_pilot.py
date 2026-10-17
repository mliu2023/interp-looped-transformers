#!/usr/bin/env python3
"""Render a compact overview from verified pilot summaries, without a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.run_dir/'summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'completed_exploratory_pilot'
    invocations = list(range(1, summary['invocations']+1))
    correlations = summary['median_l1_to_pass_spearman']
    coverage = summary['both_sign_coverage']
    assert [row['invocation'] for row in coverage] == invocations
    assert len(correlations) == len(invocations)
    units = summary['units']
    layers = summary['capture_validation']['layers']
    low, high = summary['median_l1_l4_spearman_family_bootstrap_95_interval']

    # Keep plotting caches out of the shared home directory.
    with tempfile.TemporaryDirectory(prefix='poles-plot-', dir='/tmp') as cache:
        os.environ.setdefault('MPLCONFIGDIR', cache)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({
            'font.family': 'DejaVu Sans', 'font.size': 10,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.titleweight': 'bold', 'axes.titlesize': 12,
            'svg.fonttype': 'none', 'svg.hashsalt': 'loopvl-signed-pilot',
            'axes.labelcolor': '#273344', 'text.color': '#273344',
            'xtick.color': '#465266', 'ytick.color': '#465266',
        })
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        fig.subplots_adjust(left=.065, right=.98, bottom=.23, top=.77, wspace=.34)
        fig.suptitle('LoopVL pilot: neuron responses across repeated L invocations',
                     x=.065, y=.97, ha='left', fontsize=18, fontweight='bold')
        fig.text(.065, .88,
                 f"{summary['images']} synthetic images · {summary['independent_families']} paired families · "
                 f"{summary['physical_units']} sampled MLP channels · "
                 f"{summary['invocations']} L invocations", fontsize=11, color='#596579')
        for ax in axes:
            ax.grid(axis='y', color='#e5e9ee', linewidth=.8)
            ax.set_axisbelow(True)

        ax = axes[0]
        ax.set_title('A  Image rankings relative to L1', loc='left', pad=16)
        ax.plot(invocations, correlations, marker='o', color='#3068a0', linewidth=2.2)
        ax.errorbar([4], [correlations[3]],
                    yerr=[[correlations[3]-low], [high-correlations[3]]],
                    color='#182e48', capsize=5, linewidth=1.6,
                    label='95% family-bootstrap CI at L4 only')
        ax.annotate(f'L1–L4: {correlations[3]:.3f}\n95% CI: {low:.3f}–{high:.3f}',
                    xy=(4, correlations[3]), xytext=(3.35, .52), fontsize=10,
                    arrowprops={'arrowstyle': '-', 'color': '#657386'})
        ax.set(xlim=(.8, 6.2), ylim=(-.04, 1.08), ylabel='Median Spearman correlation')
        ax.set_xticks(invocations, [f'L{i}' for i in invocations])
        ax.set_xlabel('L invocation')
        ax.set_yticks([0, .25, .5, .75, 1.0])

        ax = axes[1]
        ax.set_title('B  Both signs in image means', loc='left', pad=16)
        counts = [row['units_with_both_signed_image_means'] for row in coverage]
        bars = ax.bar(invocations, counts, width=.62, color='#d38a4a')
        ax.bar_label(bars, labels=[str(value) for value in counts], padding=4, fontsize=11)
        ax.set(ylabel=f"Channels / {summary['physical_units']} sampled",
               ylim=(0, summary['physical_units']), xlabel='L invocation')
        ax.set_xticks(invocations, [f'L{i}' for i in invocations])
        ax.set_yticks([0, 12, 24, 36, 48])

        ax = axes[2]
        ax.set_title('C  L1–L4 rankings by layer', loc='left', pad=16)
        rng = np.random.default_rng(0)  # Cosmetic jitter only.
        colors = ['#3068a0', '#5f8c7b', '#8b6b9d']
        for x, (layer, color) in enumerate(zip(layers, colors)):
            values = [row['l1_l4_spearman'] for row in units
                      if row['physical_layer'] == layer and row['l1_l4_spearman'] is not None]
            median = statistics.median(values)
            assert abs(median-summary['median_l1_l4_spearman_by_layer'][str(layer)]) < 1e-12
            ax.scatter(x+rng.uniform(-.14, .14, len(values)), values,
                       s=29, alpha=.85, color=color, edgecolors='white', linewidth=.35)
            ax.hlines(median, x-.25, x+.25, color='#182e48', linewidth=2.6)
            ax.text(x, .83, f'{median:.3f}', ha='center', fontsize=10)
        ax.axhline(0, color='#8995a5', linewidth=.8)
        ax.set(xlim=(-.55, len(layers)-.45), ylim=(-1, 1),
               ylabel='Per-channel Spearman correlation', xlabel='Physical L layer')
        ax.set_xticks(range(len(layers)), [str(layer) for layer in layers])
        ax.set_yticks([-1, -.5, 0, .5, 1])
        ax.text(.5, .035, 'Dots: channels · Bars and numbers: medians',
                transform=ax.transAxes, ha='center', fontsize=8.5, color='#596579')

        fig.text(.065, .09,
                 'Responses are means of MLP products over visual tokens. '
                 'Both-sign support requires at least four distinct families per sign.',
                 fontsize=9, color='#596579')
        fig.text(.065, .055,
                 'Error bar: 95% family-bootstrap interval at L4 only, '
                 'conditional on this checkpoint and channel sample.',
                 fontsize=9, color='#596579')
        fig.text(.065, .02,
                 'Exploratory results on this image and channel sample; '
                 'signed activity and ranking changes do not establish semantic poles.',
                 fontsize=9, color='#596579')
        stem = args.run_dir/'results_overview'
        fig.savefig(stem.with_suffix('.png'), dpi=160, facecolor='white')
        fig.savefig(stem.with_suffix('.svg'), facecolor='white', metadata={'Date': None})
        plt.close(fig)
        metadata = {'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'matplotlib_version': matplotlib.__version__,
                    'confidence_interval_scope': 'L1/L4 median only; resampled whole image families',
                    'scatter_jitter_seed': 0, 'model_forwards': 0,
                    'files': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (stem.with_suffix('.png'), stem.with_suffix('.svg'))}}
        stem.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(json.dumps({'png': str(stem.with_suffix('.png')),
                          'svg': str(stem.with_suffix('.svg'))}, indent=2))


if __name__ == '__main__':
    main()
