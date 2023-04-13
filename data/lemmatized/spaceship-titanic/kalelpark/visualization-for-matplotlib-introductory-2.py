import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]
(fig, ax) = plt.subplots(1, 3, figsize=(9, 3))
ax[0].bar(names, values)
ax[0].grid(True)
ax[1].scatter(names, values)
ax[2].plot(names, values)
ax[2].grid(True)
fig.suptitle('Category Plotting', fontsize=15)
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, values)
plt.grid(True)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.grid(True)
plt.plot(names, values)
plt.suptitle('Categorical Plotting', fontsize=15)
np.random.seed(19680801)
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))
plt.figure()
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthresh=0.01)
plt.title('symlog')
plt.grid(True)
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
plt.subplots_adjust(top=0.35, bottom=-1, left=0.1, right=0.95, hspace=0.55, wspace=0.35)
data = {'Barton LLC': 109438.5, 'Frami, Hills and Schmidt': 103569.59, 'Fritsch, Russel and Anderson': 112214.71, 'Jerde-Hilpert': 112591.43, 'Keeling LLC': 100934.3, 'Koepp Ltd': 103660.54, 'Kulas Inc': 137351.96, 'Trantow-Barrows': 123381.38, 'White-Trantow': 135841.99, 'Will LLC': 104437.6}
group_data = list(data.values())
group_names = list(data.keys())
group_mean = np.mean(group_data)
print(plt.style.available)
plt.style.use('Solarize_Light2')
(fig, ax) = plt.subplots()
(fig, ax) = plt.subplots()
ax.barh(group_names, group_data)
(fig, ax) = plt.subplots()
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
(fig, ax) = plt.subplots(figsize=(8, 4))
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax.set(xlim=[-10000, 140000], xlabel='Total Revenue', ylabel='Company', title='Company Revenue')

def currency(x, pos):
    """The two arguments are the value and tick position"""
    if x >= 1000000.0:
        s = '${:1.1f}M'.format(x * 1e-06)
    else:
        s = '${:1.0f}K'.format(x * 0.001)
    return s
(fig, ax) = plt.subplots(figsize=(10, 8))
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax.axvline(group_mean, ls='--', color='r')
for group in [3, 5, 8]:
    ax.text(145000, group, 'New Company', fontsize=10, verticalalignment='center')
ax.title.set(y=1.05)
ax.set(xlim=[-10000, 140000], xlabel='Total Revenue', ylabel='Company', title='Company Revenue')
ax.xaxis.set_major_formatter(currency)
ax.set_xticks([0, 25000.0, 50000.0, 75000.0, 100000.0, 125000.0])
(fig, ax) = plt.subplots(figsize=(10, 6))
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '--'
data = np.random.randn(50)
plt.plot(data, color='blue')
(fig, ax) = plt.subplots(figsize=(10, 6))
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.linestyle'] = '--'
data = np.random.randn(50)
plt.plot(data, color='red')
with plt.style.context('dark_background'):
    plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'r-o')