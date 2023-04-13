"""
Created on Wed Apr 29 15:53:02 2020

@author: hp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pima_main = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', na_values=['??', '????'])
pima_work = pima_main.copy(deep=True)
print(pima_work.head())
print(pima_work.shape)
pima_work.describe()
pima_main.info()
pima_work.isnull().sum()
pima_work.isin([0]).sum()
pima_work.Outcome.value_counts()
pima_nan = pima_work.replace({'Glucose': 0, 'BloodPressure': 0, 'SkinThickness': 0, 'BMI': 0, 'Insulin': 0}, np.NaN)
pima_nan.isin([0]).sum()
pima_nan.mean()
pima_nan.median()
pima_nan.isnull().sum()
pima_nan = pima_nan.fillna(pima_nan.mean())
pima_nan.isnull().sum()
from sklearn.utils import shuffle
pima_nan = shuffle(pima_nan)
pima_nan.groupby('Outcome').mean().transpose()
pima_corr = pima_nan.corr()
pass
pass
for cols in pima_nan.columns:
    x = pima_nan.loc[:, cols]
    pass
    pass
    X = pima_nan.drop('Outcome', axis=1)
    y = pima_nan['Outcome'].values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=60)
from sklearn.model_selection import cross_val_score
accuraccies = cross_val_score(estimator=KNN, X=X_train, y=y_train, cv=10)
print('Average Accuracies: ', np.mean(accuraccies))
print('Standart Deviation Accuracies: ', np.std(accuraccies))