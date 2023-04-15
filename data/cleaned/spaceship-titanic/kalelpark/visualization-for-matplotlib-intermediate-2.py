import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
plt.style.use('Solarize_Light2')

def example_plot(ax, fontsize=12, hide_labels=False):
    ax.plot([1, 2])
    if hide_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
(fig, ax) = plt.subplots(constrained_layout=False)
example_plot(ax)

(fig, axs) = plt.subplots(2, 2, constrained_layout=False)
for ax in axs.flat:
    example_plot(ax)

(fig, axs) = plt.subplots(2, 2, constrained_layout=True)
for ax in axs.flat:
    example_plot(ax)

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
arr = np.arange(100).reshape((10, 10))
norm = mcolors.Normalize(vmin=0.0, vmax=100.0)
pc_kwargs = {'rasterized': True, 'cmap': 'viridis', 'norm': norm}
(fig, ax) = plt.subplots(figsize=(4, 4), constrained_layout=True)
im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=ax, shrink=0.6)

(fig, axs) = plt.subplots(2, 2, figsize=(4, 4), constrained_layout=True)
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=axs, shrink=0.6)

(fig, axs) = plt.subplots(3, 3, figsize=(4, 4), constrained_layout=True)
count = 0
for ax in axs.flat:
    if count == 0:
        ax.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], color='red')
        count += 1
        continue
    im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=axs, shrink=0.6)

(fig, axs) = plt.subplots(3, 3, figsize=(4, 4), constrained_layout=True)
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=axs[0:,][:, 1], shrink=0.8)
fig.colorbar(im, ax=axs[:, -1], shrink=0.4)

(fig, ax) = plt.subplots(constrained_layout=True)
ax.plot(np.arange(10), label='This is plot')
ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
fig = plt.figure()
gs1 = gridspec.GridSpec(2, 1, figure=fig)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
example_plot(ax1)
example_plot(ax2)

fig = plt.figure(constrained_layout=True)
gs0 = fig.add_gridspec(1, 2)
gs1 = gs0[0].subgridspec(2, 1)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
example_plot(ax1)
example_plot(ax2)
gs2 = gs0[1].subgridspec(3, 1)
for ss in gs2:
    ax = fig.add_subplot(ss)
    example_plot(ax)
    ax.set_title('')
    ax.set_xlabel('')
ax.set_xlabel('X_label', fontsize=14)

fig = plt.figure(constrained_layout=True)
gs0 = fig.add_gridspec(1, 2, figure=fig, width_ratios=[1.0, 2.0])
gsl = gs0[0].subgridspec(2, 1)
gsr = gs0[1].subgridspec(2, 2)
for gs in gsl:
    ax = fig.add_subplot(gs)
    example_plot(ax)
axs = []
for gs in gsr:
    ax = fig.add_subplot(gs)
    pcm = ax.pcolormesh(arr, **pc_kwargs)
    ax.set_xlabel('x_label')
    ax.set_ylabel('y_label')
    ax.set_title('title')
    axs += [ax]
fig.colorbar(pcm, ax=axs)

fig = plt.figure(constrained_layout=True)
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 3)
ax3 = plt.subplot(2, 2, (2, 4))
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
