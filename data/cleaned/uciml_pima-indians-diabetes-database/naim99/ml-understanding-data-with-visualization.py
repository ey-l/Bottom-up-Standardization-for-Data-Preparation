from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas import read_csv
data = read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.hist(figsize=(10, 10), color='#5F9EA0')

plt.savefig('fig_1.png')
data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, figsize=(10, 10))

plt.savefig('fig_2.png')
data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(10, 10))

from matplotlib import pyplot
from pandas import read_csv
import numpy
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
names = data.columns.tolist()
correlations = data.corr()
fig = pyplot.figure(figsize=(22, 22))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()
from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix
scatter_matrix(data, figsize=(20, 20))
pyplot.show()