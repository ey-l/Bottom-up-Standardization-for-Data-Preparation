import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
plt.style.use('Solarize_Light2')
(fig, axs) = plt.subplots(ncols=2, nrows=2, figsize=(5.5, 3.5), constrained_layout=True)
for row in range(2):
    for col in range(2):
        axs[row, col].annotate(f'axs[{row}, {col}]', (0.5, 0.5), transform=axs[row, col].transAxes, ha='center', va='center', fontsize=18, color='darkgrey')
fig.suptitle('plt.subplots()')

def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center', fontsize=fontsize, color='darkgrey')
(fig, axd) = plt.subplot_mosaic([['upper_left', 'upper_right'], ['lower left', 'upper_right']], figsize=(5.5, 3.5), constrained_layout=True)
for k in axd:
    annotate_axes(axd[k], f'axd[{k}]', fontsize=14)
fig.suptitle('plt.suptitle_mosaic()')
gs_kw = dict(width_ratios=[1.4, 1], height_ratios=[1, 2])
(fig, axd) = plt.subplot_mosaic([['upper left', 'right'], ['lower left', 'right']], gridspec_kw=gs_kw, figsize=(5.5, 3.5), constrained_layout=True)
for k in axd:
    annotate_axes(axd[k], f'axd[{k}]', fontsize=14)
fig.suptitle('plt.subplot_mosaic()')
fig = plt.figure(constrained_layout=True)
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1.5, 1.0])
axs0 = subfigs[0].subplots(2, 2)
subfigs[0].set_facecolor('0.9')
subfigs[0].suptitle('subfigs[0]\nLeft side')
subfigs[0].supxlabel('xlabel for subfigs[0]')
axs1 = subfigs[1].subplots(3, 1)
subfigs[1].suptitle('supfig[1]')
subfigs[1].supylabel('ylabel for subfigs[1]')
inner = [['innerA'], ['innerB']]
outer = [['upper left', inner], ['lower left', 'lower right']]
(fig, axd) = plt.subplot_mosaic(outer, constrained_layout=True)
for k in axd:
    annotate_axes(axd[k], f'axd[{k}]')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sinc(x)
(fig, ax) = plt.subplots()
ax.plot(x, y)
(fig, ax) = plt.subplots()
ax.plot(x, y)
ax.margins(0.2, 0.2)
(xx, yy) = np.meshgrid(x, x)
zz = np.sinc(np.sqrt((xx - 1) ** 2 + (yy - 1) ** 2))
(fig, ax) = plt.subplots(1, 2, figsize=(12, 8))
ax[0].imshow(zz)
ax[0].set_title('default margin')
ax[1].imshow(zz)
ax[1].margins(0.5)
ax[1].set_title('margin(0.2)')
(fig, ax) = plt.subplots(1, 2, figsize=(12, 8))
ax[0].plot(x, y)
ax[0].set_title('single curve')
ax[1].plot(x, y)
ax[1].plot(x * 2, y, color='orange')
ax[1].set_title('two curve')
(fig, ax) = plt.subplots(1, 2, figsize=(12, 8))
ax[0].plot(x, y)
ax[0].set_xlim(left=-1, right=1)
ax[0].plot(x + np.pi * 0.5, y, color='orange')
ax[1].plot(x, y)
ax[1].set_xlim(left=-1, right=1)
ax[1].plot(x + np.pi * 0.5, y, color='orange')
ax[1].autoscale()
ax[1].set_title('set_xlim(left=-1, right=1)\nautoscale()')
(fig, ax) = plt.subplots()
collection = mpl.collections.StarPolygonCollection(5, 0, [250], offsets=np.column_stack([x, y]), transOffset=ax.transData)
ax.add_collection(collection)
ax.autoscale_view()