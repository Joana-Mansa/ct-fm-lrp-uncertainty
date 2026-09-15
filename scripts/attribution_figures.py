"""Display saved attribution maps with verified test indices and fixed colour scales.

This script renders existing arrays; it does not compute new attributions.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / 'docs' / 'figures'
METHODS = [
    ('gradcam', 'Grad-CAM'),
    ('epsilon_plus_flat', 'Zennit LRP\nEpsilonPlusFlat'),
    ('epsilon_gamma_box', 'Zennit LRP\nEpsilonGammaBox'),
    ('epsilon_alpha2_beta1', 'Zennit LRP\nEpsilonAlpha2Beta1'),
]


def render():
    path = ROOT / 'results' / 'qualitative.npz'
    info = json.loads((ROOT / 'results' / 'attribution_examples.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
        raise ValueError('Saved maps changed; recheck test identities before rendering.')
    with np.load(path) as z:
        for key, _ in METHODS:
            if not np.isfinite(z[key]).all() or z[key].min() < 0 or z[key].max() > 1:
                raise ValueError(f'Expected finite, normalised saved map: {key}')
        if not np.array_equal(z['label'], info['labels']) or not np.array_equal(z['pred'], info['predictions']):
            raise ValueError('Labels/predictions disagree with verified example metadata.')
        for n, name in [(2, 'attribution_preview'), (4, 'attribution_panel')]:
            height = 2.55*n + 1.8
            fig, axes = plt.subplots(n, 5, figsize=(12.8, height), squeeze=False)
            fig.subplots_adjust(left=.015, right=.985, top=1-1.15/height, bottom=1.25/height, hspace=.32, wspace=.055)
            fig.suptitle('Real CT inputs and saved attribution maps', y=1-.08/height, fontsize=18, weight='bold', color='#203448')
            fig.text(.5, 1-.55/height, 'Target = predicted class  |  Same central slice in every column  |  Seed-0 CT-FM classifier', ha='center', fontsize=11, color='#566777')
            for col, title in enumerate(['CT input'] + [title for _, title in METHODS]):
                axes[0, col].set_title(title, fontsize=11, pad=12)
            for row in range(n):
                volume = z['volume'][row, 0]
                mid = info['slice_index']
                for col in range(5):
                    ax = axes[row, col]
                    ax.imshow(volume[mid], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
                    if col:
                        ax.imshow(z[METHODS[col-1][0]][row, mid], cmap='magma', vmin=0, vmax=1, alpha=.55, interpolation='nearest')
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values(): spine.set_visible(False)
                target = info['predictions'][row]
                prob = info['checkpoint_probabilities'][row][target]
                text = f"Test {info['test_indices'][row]} | label {info['labels'][row]} | prediction {target}\npredicted-class probability {prob:.4f}"
                axes[row, 0].set_xlabel(text, fontsize=9, labelpad=7)
            bar_ax = fig.add_axes([.37,.55/height,.35,.13/height])
            cb = fig.colorbar(ScalarMappable(norm=Normalize(0,1), cmap='magma'), cax=bar_ax, orientation='horizontal')
            cb.set_ticks([0, .5, 1]); cb.ax.tick_params(labelsize=9)
            fig.text(.32,.615/height,'Low',ha='right',va='center',fontsize=10)
            fig.text(.75,.615/height,'High',ha='left',va='center',fontsize=10)
            fig.text(.5,.045/height,'Per-volume min-max normalisation; magnitude and sign cannot be compared across methods.\nLabel 0 / 1: lower / higher malignancy ratings. Array axis 0, slice 32. Maps are exploratory.',ha='center',fontsize=10,color='#566777')
            FIGS.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGS/f'{name}.png', dpi=180, facecolor='white')
            plt.close(fig)


if __name__ == '__main__':
    render()
