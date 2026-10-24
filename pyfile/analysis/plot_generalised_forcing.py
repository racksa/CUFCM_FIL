"""Reference generalised forcing over one beat cycle for two beat patterns.

Plots the phase and tilt (shape-rotation) generalised forces recorded by the
calibration runs (WRITE_GENERALISED_FORCES) for the Fulford-Blake beat
(SHAPE_SEQUENCE=1) and the MLS beat (SHAPE_SEQUENCE=8).

Forces are converted from code units (mu = 1, T = 1, lengths in segment radii)
to units of mu L^3 / T by dividing by L^3, with L the filament length.

Run from the repository root:  python3 pyfile/analysis/plot_generalised_forcing.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_dir = os.path.expanduser("~/.local/share/fonts/cmu/cm-unicode-0.7.0")
if os.path.isdir(font_dir):
    for font_file in os.listdir(font_dir):
        if font_file.endswith('.otf'):
            fm.fontManager.addfont(os.path.join(font_dir, font_file))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['CMU Serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 16})

FIL_LENGTH = 49.4          # segment radii: DL*(NSEG-1) = 2.6*19
SUFFIX = '_NSEG=20_SEP=2.600000.dat'
BEATS = [
    # label, file stem, colour (reference palette slots 1 and 2), line style
    ('Fulford–Blake (SS1)', 'fulford_and_blake', '#2a78d6', '-'),
    ('MLS (SS8)',           'mls_precomputed',   '#eb6834', '--'),
]
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'


def load(stem, kind):
    path = f'input/forcing/{stem}_reference_{kind}_generalised_forces{SUFFIX}'
    values = np.loadtxt(path).ravel()
    n = int(values[0])                      # first entry is the number of samples
    return values[1:n + 1] / FIL_LENGTH**3


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
panels = [('phase', r'$Q_1\;[\mu L^3/T]$', 'Phase forcing'),
          ('angle', r'$Q_2\;[\mu L^3/T]$', 'Angle forcing')]

for ax, letter, (kind, ylabel, title) in zip(axes, ('(a)', '(b)'), panels):
    curves = []
    for label, stem, colour, style in BEATS:
        q = load(stem, kind)
        phase = np.arange(len(q)) / len(q)
        ax.plot(phase, q, style, color=colour, linewidth=2, label=label)
        curves.append((label.split(' (')[0], phase, q))
    # Direct label for each curve, placed where it sits furthest above the other
    # curve, so the two labels never collide.
    (name_a, x_a, q_a), (name_b, x_b, q_b) = curves
    q_b_on_a = np.interp(x_a, x_b, q_b)
    for name, x, q, gap in ((name_a, x_a, q_a, q_a - q_b_on_a), (name_b, x_a, q_b_on_a, q_b_on_a - q_a)):
        k = int(np.argmax(np.where((x > 0.08) & (x < 0.92), gap, -np.inf)))
        right = x[k] < 0.5            # put the label on the side facing the plot interior
        ax.annotate(name, xy=(x[k], q[k]), xytext=(10 if right else -10, 4), textcoords='offset points',
                    ha='left' if right else 'right', va='bottom', fontsize=12, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title(title, color=INK, fontsize=16)
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, ha='right', va='bottom', fontsize=15, color=INK)
    ax.set_xlabel(r'$\psi_1 / 2\pi$', color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xlim(0, 1)
    ax.margins(y=0.12)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=13, loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(w_pad=3)
os.makedirs('fig', exist_ok=True)
fig.savefig('fig/generalised_forcing_SS1_vs_SS8.png', dpi=200, bbox_inches='tight')
fig.savefig('fig/generalised_forcing_SS1_vs_SS8.pdf', bbox_inches='tight')
print('saved fig/generalised_forcing_SS1_vs_SS8.png and .pdf')
plt.show()
