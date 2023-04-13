import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
plt.style.use('Solarize_Light2')
verts = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.3), (1.0, 0.0), (0.0, 0.0)]
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
path = Path(verts, codes)
(fig, ax) = plt.subplots()
patch = patches.PathPatch(path, facecolor='orange', lw=2)
ax.add_patch(patch)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
verts = [(0.0, 0.0), (0.2, 1.0), (1.0, 0.8), (0.8, 0.0)]
codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
path = Path(verts, codes)
(fig, ax) = plt.subplots()
patch = patches.PathPatch(path, facecolor='none', lw=2)
ax.add_patch(patch)
(xs, ys) = zip(*verts)
ax.plot(xs, ys, 'x--', lw=2, color='black', ms=10)
ax.text(-0.05, -0.05, 'P0')
ax.text(0.15, 1.05, 'P1')
ax.text(1.05, 0.85, 'P2')
ax.text(0.85, -0.05, 'P3')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
fig = plt.figure(figsize=(7, 7))
text = fig.text(0.5, 0.5, 'Hello Path effects world \n this is the normal Patch effect. \n pretty dull, huh?', ha='center', va='center', size=20)
plt.plot([], path_effects=path_effects.Normal())
import matplotlib.textpath as patch_effects
text = plt.text(0.5, 0.5, 'Hello path effects world!', path_effects=[path_effects.withSimplePatchShadow()], fontsize=20)
plt.plot([0, 3, 2, 5], linewidth=5, color='blue', path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
x = np.arange(0, 10, 0.005)
y = np.exp(-x / 2.0) * np.sin(2 * np.pi * x)
(fig, ax) = plt.subplots()
ax.plot(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)
x = np.arange(0, 10, 0.005)
y = np.exp(-x / 2.0) * np.sin(2 * np.pi * x)
(fig, ax) = plt.subplots()
ax.plot(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)
(xdata, ydata) = (5, 0)
(xdisplay, ydisplay) = ax.transData.transform((xdata, ydata))
bbox = dict(boxstyle='round', fc='0.8')
arrowprops = dict(arrowstyle='->', connectionstyle='angle, angleA = 0, angleB = 90, rad =10')
offset = 72
ax.annotate('data = (%.1f, %.1f)' % (xdata, ydata), (xdata, ydata), xytext=(-2 * offset, offset), textcoords='offset points', bbox=bbox, arrowprops=arrowprops)
disp = ax.annotate('data = (%.1f, %.1f)' % (xdata, ydata), (xdisplay, ydisplay), xytext=(0.5 * offset, -offset), xycoords='figure pixels', textcoords='offset points', bbox=bbox, arrowprops=arrowprops)