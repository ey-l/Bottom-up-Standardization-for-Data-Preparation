import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot
from pandas import read_csv
import numpy as np
import pandas as pd
import numpy
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix
path = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
data = pd.read_csv(path)
data.head()
print(data.shape)
print(data.dtypes)
data.info(verbose=True)
print(data.shape)
print(data.describe())
count_Outcome = data.groupby('Outcome').size()
print(count_Outcome)
correlations = data.corr(method='pearson')
print(correlations)
print(data.skew())
data.hist()
pyplot.show()
data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
pyplot.show()
data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
pyplot.show()
fig = pyplot.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
pyplot.show()
scatter_matrix(data)
pyplot.show()
from numpy import set_printoptions
from sklearn import preprocessing
array = data.values
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_rescaled = data_scaler.fit_transform(array)
set_printoptions(precision=1)
print('\nScaled data:\n', data_rescaled[0:10])
from sklearn.preprocessing import Normalizer