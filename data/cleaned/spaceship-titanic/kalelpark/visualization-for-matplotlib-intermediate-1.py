import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
plt.style.use('Solarize_Light2')
fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(211)
ax1.set_ylabel('volts')
ax1.set_title('a sine wave')
t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2 * np.pi * t)
(line,) = ax1.plot(t, s, color='red', lw=2)
ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.3])
(n, bins, patches) = ax2.hist(np.random.randn(1000), 50, facecolor='blue', edgecolor='yellow')
ax2.set_xlabel('time (s)')

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
plt.style.use('Solarize_Light2')
(fig, ax) = plt.subplots(2, 1, figsize=(8, 5))
ax[0].plot(t, s, color='red', lw=2)
ax[0].set_ylabel('volts')
ax[0].set_title('a sine wave')
ax[1].hist(np.random.randn(1000), 50, facecolor='blue', edgecolor='yellow')
ax[1].set_xlabel('time (s)')

fig = plt.figure()
fig.patch.set_facecolor('#a5e6b6')
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes1.patch.set_facecolor('white')
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
axes2.patch.set_facecolor('white')
fig = plt.figure()
fig.patch.set_facecolor('lightgoldenrodyellow')
axes1 = fig.add_axes([0.1, 0.1, 0.9, 0.9])
axes1.plot(t, s)
axes1.patch.set_facecolor('lightgoldenrodyellow')
axes2 = fig.add_axes([0.2, 0.4, 0.5, 0.5])
axes2.plot(t, s, color='blue')
axes2.patch.set_facecolor('red')
axes2.grid(False)
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(19680801)
(fig, ax) = plt.subplots()
ax.plot(100 * np.random.rand(20))
ax.yaxis.set_major_formatter('${x:1.2f}')
ax.yaxis.set_tick_params(which='major', labelcolor='green', labelleft=False, labelright=True)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
(fig, ax) = plt.subplots()
red_patch = mpatches.Patch(color='red', label='The red data')
ax.legend(handles=[red_patch])

import matplotlib.lines as mlines
(fig, ax) = plt.subplots()
blue_line = mlines.Line2D([], [], color='blue', marker='*', markersize=15, label='Blue Star')
ax.legend(handles=[blue_line])

ax.legend(bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
(fig, ax_dict) = plt.subplot_mosaic([['top', 'top'], ['bottom', 'blank']], empty_sentinel='blank', figsize=(10, 10))
ax_dict['top'].plot([1, 2, 3], label='test1')
ax_dict['top'].plot([3, 2, 1], label='test2')
ax_dict['top'].legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc='lower left', ncol=2, mode='expand', borderaxespad=0.0)
ax_dict['bottom'].plot([1, 2, 3], label='test1')
ax_dict['bottom'].plot([3, 2, 1], label='test2')
ax_dict['bottom'].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)

(fig, ax) = plt.subplots()
(line1,) = ax.plot([1, 2, 3], label='Line1', linewidth=4)
(line2,) = ax.plot([3, 2, 1], label='Line2', linestyle='--', linewidth=4)
first_legend = ax.legend(handles=[line1], loc='upper right')
ax.add_artist(first_legend)
second_legend = ax.legend(handles=[line2], loc='lower right')
ax.add_artist(second_legend)
ax.set_facecolor('white')

z = np.random.randn(10)
(fig, ax) = plt.subplots()
(red_dot,) = ax.plot(z, 'ro', markersize=15)
(white_cross,) = ax.plot(z[:5], 'w+', markeredgewidth=3, markersize=15)
ax.legend([red_dot, (red_dot, white_cross)], ['Attr A', 'Attr A+B'])

from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2 * np.pi, 50)
offsets = np.linspace(0, 2 * np.pi, 4, endpoint=False)
yy = np.transpose([np.sin(x + phi) for phi in offsets])
default_cycler = cycler(color=['r', 'g', 'b', 'y']) + cycler(linestyle=['-', '--', ':', '-.']) + cycler(lw=[1, 2, 3, 4])
default_cycler1 = cycler(color=['r', 'g', 'b', 'y']) + cycler(lw=[1, 2, 3, 4])
plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=default_cycler)
(fig, ax) = plt.subplots(2, 1, figsize=(10, 7))
ax[0].set_prop_cycle(default_cycler1)
ax[0].plot(yy)
ax[0].set_title('Set default cycle to rgby')
ax[1].set_prop_cycle(default_cycler)
ax[1].plot(yy)
ax[1].set_title('Set axes color cycle to cmyk')
