import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(_input1.dtypes)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
for i in list(_input1.select_dtypes(include=['float64'])):
    plt.plot(_input1[i])
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in list(_input1.select_dtypes(include=['object'])):
    _input1[i] = le.fit_transform(_input1[i].ravel())
    _input0[i] = le.fit_transform(_input0[i].ravel())
_input1.dtypes
model = XGBClassifier()