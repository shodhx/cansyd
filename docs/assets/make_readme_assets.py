"""README visual assets for CNSD. Modern, color, GitHub-friendly (renders on
light and dark). PNG at 2x for crispness."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update(
    {
        'font.family': 'DejaVu Sans',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        'savefig.dpi': 200,
    }
)

# palette
INK = '#0f172a'  # slate-900
SLATE = '#475569'  # slate-600
BLUE = '#2563eb'  # blue-600
TEAL = '#0d9488'  # teal-600
AMBER = '#d97706'  # amber-600
GREEN = '#16a34a'  # green-600
RED = '#dc2626'  # red-600
VIOL = '#7c3aed'  # violet-600
BG = '#f8fafc'


# ---------------------------------------------------------------------------
# HERO BANNER
# ---------------------------------------------------------------------------
def hero(path):
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 34)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 120, 34, fc=BG, ec='none', zorder=0))
    # left accent bar
    ax.add_patch(plt.Rectangle((0, 0), 1.4, 34, fc=BLUE, ec='none', zorder=1))

    ax.text(6, 22.5, 'CNSD', fontsize=54, fontweight='bold', color=INK, va='center', zorder=3)
    ax.text(
        6.4,
        12.8,
        'Causal Neuro-Symbolic Diagnosis',
        fontsize=15.5,
        color=BLUE,
        fontweight='bold',
        va='center',
    )
    ax.text(
        6.6,
        7.0,
        "Verify \u2014 don't just predict \u2014 bearing faults",
        fontsize=11,
        color=SLATE,
        va='center',
    )
    ax.text(6.6, 3.2, 'under operating-condition shift.', fontsize=11, color=SLATE, va='center')

    # right: waveform -> envelope comb motif
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 600)
    burst = np.zeros_like(x)
    for c in np.arange(0.05, 1, 0.11):
        burst += np.exp(-((x - c) ** 2) / 2e-4) * np.sin(2 * np.pi * 90 * (x - c))
    burst += 0.15 * rng.standard_normal(x.size)
    xa, xb = 74, 116
    ax.plot(
        np.linspace(xa, xb, x.size),
        26 + 3.2 * burst / np.abs(burst).max(),
        color=SLATE,
        lw=0.8,
        zorder=2,
    )
    ax.text(xa, 30.6, 'vibration', fontsize=7.5, color=SLATE)
    # comb
    combx = np.linspace(xa, xb, 9)
    for i, cx in enumerate(combx):
        h = [10, 3, 6, 2, 4, 1.5, 3, 1, 2][i]
        col = BLUE if i in (0, 2, 4) else '#cbd5e1'
        ax.add_patch(plt.Rectangle((cx - 0.5, 14), 1.0, h, fc=col, ec='none', zorder=2))
    ax.text(
        xa, 12.2, 'envelope spectrum  →  characteristic-frequency comb', fontsize=7.5, color=SLATE
    )

    # verdict chips (lower-right, correctly sized)
    chips = [('CONFIRMED', GREEN), ('CONFLICT', RED), ('INCONCLUSIVE', AMBER)]
    cx = 74
    for name, col in chips:
        w = 0.44 * len(name) + 2.4
        ax.add_patch(
            FancyBboxPatch(
                (cx, 2.4),
                w,
                4.0,
                boxstyle='round,pad=0.15,rounding_size=1.4',
                fc='white',
                ec=col,
                lw=1.5,
                zorder=3,
            )
        )
        ax.text(
            cx + w / 2,
            4.4,
            name,
            ha='center',
            va='center',
            fontsize=6,
            color=col,
            fontweight='bold',
            zorder=4,
        )
        cx += w + 1.4

    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ARCHITECTURE (colored 5-layer)
# ---------------------------------------------------------------------------
def architecture(path):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 116)
    ax.set_ylim(0, 60)
    ax.axis('off')

    def box(x, y, w, h, title, sub, accent, fc='white'):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle='round,pad=0.4,rounding_size=1.6',
                fc=fc,
                ec=accent,
                lw=2.0,
                zorder=3,
            )
        )
        ax.add_patch(
            plt.Rectangle((x + 0.6, y + 0.6), 1.1, h - 1.2, fc=accent, ec='none', zorder=4)
        )
        ax.text(
            x + w / 2 + 0.6,
            y + h - 3.4,
            title,
            ha='center',
            va='top',
            fontsize=9.5,
            fontweight='bold',
            color=INK,
            zorder=5,
        )
        ax.text(
            x + w / 2 + 0.6,
            y + h - 8.2,
            sub,
            ha='center',
            va='top',
            fontsize=7.2,
            color=SLATE,
            linespacing=1.3,
            zorder=5,
        )

    def arrow(x1, y1, x2, y2, color=SLATE, rad=0.0, lw=2.0, ls='-'):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle='-|>',
                mutation_scale=15,
                lw=lw,
                color=color,
                zorder=2,
                ls=ls,
                connectionstyle=f'arc3,rad={rad}',
            )
        )

    box(1, 22, 16, 13, 'Vibration\nwindow', 'raw signal\nx  \u2208  \u211d^L', SLATE, fc='#eef2f7')
    box(
        21,
        21,
        19,
        15,
        '1 · Perception',
        '1-D CNN / S-JEPA\nproposes fault \u0177\n+ confidence c',
        VIOL,
    )
    box(
        46,
        39,
        24,
        15,
        '2 · Symbolic',
        'envelope-spectrum check\nCONFIRMED / CONFLICT /\nINCONCLUSIVE + root cause',
        TEAL,
    )
    box(
        46,
        21.5,
        24,
        14,
        '3 · Causal (Rung 2)',
        'do(Z) on corrected SCM\ninterventional warrant\n+ refutation suite',
        BLUE,
    )
    box(
        46,
        3.5,
        24,
        14,
        '3B · Counterfactual',
        'invertible SCM, RMS\nRung-3 stability\nunder do(Z:=z\u2032)',
        AMBER,
    )
    box(76, 21, 18, 15, '4 · Consensus', 'fuse verdict + c\nphysics can VETO\nthe network', INK)
    box(98, 21.5, 16, 14, 'Decision', 'HIGH_CONF\nRELIABLE\nUNCERTAIN\nREVIEW', GREEN, fc='#f0fdf4')

    arrow(17, 28.5, 21, 28.5)
    arrow(40, 28.5, 46, 46, color=TEAL, rad=0.15)
    arrow(40, 28.5, 46, 28.5, color=BLUE)
    arrow(40, 28.5, 46, 10.5, color=AMBER, rad=-0.15)
    arrow(70, 46, 76, 30, color=TEAL, rad=0.12)
    arrow(70, 28.5, 76, 28.5, color=BLUE)
    arrow(70, 10.5, 76, 27, color=AMBER, rad=-0.12)
    arrow(94, 28.5, 98, 28.5, color=GREEN)
    # veto
    arrow(60, 39, 44, 33, color=RED, rad=0.3, lw=1.6, ls=(0, (4, 2)))
    ax.text(49.5, 38.4, 'veto', fontsize=7.5, color=RED, style='italic', fontweight='bold')

    ax.text(
        58,
        58,
        'propose  →  verify  →  decide',
        ha='center',
        fontsize=10,
        color=SLATE,
        style='italic',
    )
    fig.savefig(path, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# RESULTS (physics vs ensemble, honest, from EXPERIMENTS.md)
# ---------------------------------------------------------------------------
def results(path):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    data = [  # dataset, physics, ens, p, regime
        ('PU\n900\u21921500 rpm', 0.538, 0.386, 'p = 0.003', 'shift'),
        ('XJTU-SY\ncross-condition', 0.460, 0.381, 'p = 0.003', 'shift'),
        ('CWRU\ncross-load', 0.189, 0.074, 'p = 0.07 (n.s.)', 'saturated'),
    ]
    x = np.arange(len(data))
    w = 0.36
    phys = [d[1] for d in data]
    ens = [d[2] for d in data]
    ax.bar(x - w / 2, phys, w, label='Physics (CNSD)', color=BLUE, zorder=3)
    ax.bar(x + w / 2, ens, w, label='Deep ensemble', color='#cbd5e1', zorder=3)
    for i, d in enumerate(data):
        ax.text(
            i,
            max(phys[i], ens[i]) + 0.03,
            d[3],
            ha='center',
            fontsize=8,
            color=(GREEN if 'n.s.' not in d[3] else SLATE),
            fontweight='bold',
        )
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in data], fontsize=9)
    ax.set_ylabel('matched-coverage separation gap  \u0394', fontsize=9.5)
    ax.set_title(
        'Physics verification vs. the strongest uncertainty baseline',
        fontsize=10.5,
        fontweight='bold',
        color=INK,
        pad=10,
    )
    ax.set_ylim(0, 0.65)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e2e8f0', zorder=0)
    ax.tick_params(labelsize=8.5)
    ax.annotate(
        'physics wins under shift',
        (0.5, 0.56),
        fontsize=8.5,
        color=GREEN,
        ha='center',
        style='italic',
    )
    ax.annotate(
        'null control:\nsaturated regime',
        (2, 0.30),
        fontsize=8,
        color=SLATE,
        ha='center',
        style='italic',
    )
    fig.savefig(path, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    import os

    os.makedirs('readme_assets', exist_ok=True)
    hero('readme_assets/hero.png')
    architecture('readme_assets/architecture.png')
    results('readme_assets/results.png')
    print('done')
