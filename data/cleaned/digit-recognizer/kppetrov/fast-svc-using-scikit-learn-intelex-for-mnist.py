import pandas as pd
import numpy as np
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
x_train = train[train.columns[1:]]
x_test = test
y_train = train[train.columns[0]]
train.head()
from sklearn.preprocessing import MinMaxScaler