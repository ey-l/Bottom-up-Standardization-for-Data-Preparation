import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
num_features = _input1.describe().columns.tolist()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(_input1[num_features], _input1['Transported'], test_size=0.3, random_state=0)
X_train.head()
X_train = X_train.fillna(X_train.median(), inplace=False)
X_test = X_test.fillna(X_train.median(), inplace=False)
X_test.head()
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()
maximum = X_train_copy['Age'].max()
minimum = X_test_copy['Age'].min()
roomRange = maximum - minimum
width = int(roomRange / 10)
width
min_value = int(np.floor(minimum))
max_value = int(np.ceil(maximum))
print(' min-->', min_value, '\n', 'max-->', max_value, '\n', 'inerval_width-->', width)
intervals = [i for i in range(min_value, max_value + width, width)]
intervals
X_train_copy['Age_Bins'] = pd.cut(x=X_train_copy['Age'], bins=intervals, include_lowest=True)
X_test_copy['Age_Bins'] = pd.cut(x=X_test_copy['Age'], bins=intervals, include_lowest=True)
X_train_copy.head()
t1 = X_train_copy['Age_Bins'].value_counts() / len(X_train)
t2 = X_test_copy['Age_Bins'].value_counts() / len(X_test)
tmp = pd.concat([t1, t2], axis=1)
tmp.columns = ['train', 'test']
tmp.plot.bar()
plt.xticks(rotation=45)
plt.ylabel('Number of observations per bin')
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()
from feature_engine.discretisation import EqualWidthDiscretiser
widthDiscretiser = EqualWidthDiscretiser(bins=10, variables=['Age'])