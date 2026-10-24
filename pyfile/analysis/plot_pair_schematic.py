"""Black-and-white schematic of the flagellar pair geometry (talk slide 8).

Two panels in one strip, sized to the slide's figure slot (12.1 x 1.4 in):
  (a) view along the beat direction, to scale: two bead chains (NSEG = 20,
      bead spacing 2.6a) standing a distance d = 5a apart on the surface;
      each beats in a plane seen edge-on
  (b) top view (not to scale): two parallel beat planes, d apart

Drawing coordinates are inches.

Run from the repository root:  python3 pyfile/analysis/plot_pair_schematic.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

plt.rcParams['font.family'] = ['Carlito', 'Calibri', 'Lato', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'cm'

W, H = 12.13, 1.40
K = 'black'
FS = 13
NSEG, DL, SEP = 20, 2.6, 5.0     # beads per filament, bead spacing and pair separation in bead radii a

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.axis('off')


def arrow(p, q, style='-|>', lw=1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=K, lw=lw,
                                 shrinkA=0, shrinkB=0))


def surface(x1, x2, y):
    """Wall line with short hatching underneath."""
    ax.plot([x1, x2], [y, y], color=K, lw=1.2)
    for x in np.arange(x1 + 0.05, x2, 0.12):
        ax.plot([x, x - 0.07], [y, y - 0.07], color=K, lw=0.6)


def label(x, y, s, **kw):
    ax.text(x, y, s, fontsize=FS, color=K, **kw)


# ---------------------------------------------------------------- (a) along the beat direction, to scale
surf, xc = 0.25, 2.7
a = 0.86 / (DL*(NSEG - 1) + 1)            # inches per bead radius: whole chain fits in 0.86 in
surface(1.5, 3.9, surf)
for name, xf in (('A', xc - SEP*a/2), ('B', xc + SEP*a/2)):
    for k in range(NSEG):
        ax.add_patch(Circle((xf, surf + a*(1 + DL*k)), a, facecolor='white', edgecolor=K, lw=0.5))
top = surf + a*(2 + DL*(NSEG - 1))
label(xc - SEP*a/2 - 0.08, top - 0.02, 'A', ha='right', va='top')
label(xc + SEP*a/2 + 0.08, top - 0.02, 'B', ha='left', va='top')
# separation d, marked just above the surface
yd = surf + 0.12
arrow((xc - SEP*a/2 - 0.22, yd), (xc - SEP*a/2, yd))
arrow((xc + SEP*a/2 + 0.22, yd), (xc + SEP*a/2, yd))
label(xc + SEP*a/2 + 0.26, yd, r'$d = 5a$', va='center')
# filament length
xl = xc - 0.55
arrow((xl, surf), (xl, top), style='<|-|>')
label(xl - 0.06, (surf + top)/2, r'$L \approx 49a$', ha='right', va='center')
label(3.25, 0.80, 'beat direction\ninto the page', ha='left', va='center', linespacing=1.1)
label(0.3, 1.38, '(a)  along the beat, to scale', va='top')

# ---------------------------------------------------------------- (b) top view
bx, yA, yB = 8.0, 0.35, 0.85
for name, y in (('A', yA), ('B', yB)):
    ax.plot([bx - 1.0, bx + 1.4], [y, y], color=K, lw=0.8, ls=(0, (4, 3)))   # beat plane, edge-on
    ax.plot([bx, bx + 1.0], [y, y], color=K, lw=2.4)                         # flagellum lying in its plane
    ax.add_patch(Circle((bx, y), 0.045, facecolor='white', edgecolor=K, lw=1.1, zorder=5))
    label(bx - 1.1, y, name, ha='right', va='center')
arrow((bx + 2.65, yA), (bx + 2.65, yB), style='<|-|>')
label(bx + 2.73, (yA + yB)/2, r'$d$', va='center')
arrow((bx + 0.1, (yA + yB)/2), (bx + 0.7, (yA + yB)/2))
label(bx + 0.75, (yA + yB)/2, 'beat direction', va='center')
label(bx + 1.5, yB, 'beat planes', ha='left', va='center')
label(6.6, 1.38, '(b)  top view (not to scale)', va='top')

os.makedirs('fig', exist_ok=True)
fig.savefig('fig/pair_schematic.png', dpi=300)
fig.savefig('fig/pair_schematic.pdf')
print('saved fig/pair_schematic.png and .pdf')
