import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
training_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(training_data.head())
print(training_data.isna().sum())
print('-------------------------')
print(test_data.isna().sum())
training_data.dtypes
for col in training_data.columns:
    if training_data[col].dtypes == 'float64':
        training_data[col].fillna(training_data[col].mean(), inplace=True)
        test_data[col].fillna(test_data[col].mean(), inplace=True)
    elif training_data[col].dtypes == 'object':
        training_data[col].fillna(training_data[col].value_counts()[0], inplace=True)
        test_data[col].fillna(test_data[col].value_counts()[0], inplace=True)
print(training_data.isna().sum())
print('-----------------------')
print(test_data.isna().sum())
training_data['CryoSleep'] = training_data['CryoSleep'].astype(int)
training_data['VIP'] = training_data['VIP'].astype(int)
training_data['Transported'] = training_data['Transported'].astype(int)
test_data['CryoSleep'] = test_data['CryoSleep'].astype(int)
test_data['VIP'] = test_data['VIP'].astype(int)
print(training_data.head())
print('----------------------------------------------------------------')
print(test_data.head())
training_data = pd.get_dummies(training_data)
test_data1 = pd.get_dummies(test_data)
print(training_data.shape)
print('------------')
print(test_data1.shape)
input_features = training_data.drop('Transported', axis=1)
target = training_data['Transported']
test_data1 = test_data1.reindex(columns=input_features.columns, fill_value=0)
test_data1.shape
input_features_numpy = input_features.values
test_data_numpy = test_data1.values
min_max_scaler = MinMaxScaler()
input_features_numpy = min_max_scaler.fit_transform(input_features_numpy)
test_data_numpy = min_max_scaler.fit_transform(test_data_numpy)
print(input_features_numpy[0:5])
print('------------------------------')
print(test_data_numpy[0:5])
logreg = LogisticRegression(max_iter=1000)
kf = KFold(n_splits=5, shuffle=True, random_state=4)
cv = cross_val_score(logreg, input_features_numpy, target, cv=kf)
print(cv)