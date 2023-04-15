import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(df_train.dtypes)
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
for i in list(df_train.select_dtypes(include=['float64'])):
    plt.plot(df_train[i])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in list(df_train.select_dtypes(include=['object'])):
    df_train[i] = le.fit_transform(df_train[i].ravel())
    df_test[i] = le.fit_transform(df_test[i].ravel())
df_train.dtypes
model = XGBClassifier()