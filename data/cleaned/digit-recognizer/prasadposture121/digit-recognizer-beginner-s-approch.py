import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
data = pd.read_csv('data/input/digit-recognizer/train.csv')
pdata = pd.read_csv('data/input/digit-recognizer/test.csv')
data.head()
pdata.head()
print('Shape of train data:', data.shape)
print('Shape if we drop missing values from train data:', data.dropna(how='any').shape)
print('Shape of prediction data:', pdata.shape)
print('Shape if we drop missing values from prediction data:', pdata.dropna(how='any').shape)
a = data.iloc[3, 1:].values
a = a.reshape(28, 28).astype('uint8')
plt.imshow(a)
x = data.iloc[:, 1:]
Y = data.iloc[:, 0]
(x_train, x_test, Y_train, Y_test) = train_test_split(x, Y, test_size=0.2, random_state=4)
x_train.head()
Y_train.head()
rf = RandomForestClassifier(n_estimators=100)