import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(f'Train Data\n{_input1.head(5)}\n\nTest Data\n{_input0.head(5)}\nsubmission Data\n\n{_input2.head(5)}')
print(f'Training data Null Values :\n\n{_input1.isnull().sum()}\n\nShape :\n{_input1.shape}')
drop_col = ['Name', 'Cabin']
_input1 = _input1.drop(drop_col, axis=1, inplace=False)
_input0 = _input0.drop(drop_col, axis=1, inplace=False)
plt.subplots(1, 2, figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(_input1.isnull(), interpolation='nearest', cmap='Blues', aspect='auto')
plt.xlabel('Columns of Train data')
plt.ylabel('Values index no of the Null values')
plt.title('Heatmap for the null values')
plt.subplot(1, 2, 2)
_input1.isnull().mean().plot(kind='bar', title='Mean of the Null values', ylabel='Missing values Ratio')
_input1.info()
count_col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for col in _input1[count_col]:
    print(_input1[col].value_counts())
    print('\n')
    print(_input1[col].mode())
    print('\n--------------------------')
for col in count_col:
    _input1[col] = _input1[col].fillna(_input1[col].mode()[0], inplace=False)
    _input0[col] = _input0[col].fillna(_input0[col].mode()[0], inplace=False)
_input1.isnull().sum()
sns.violinplot(_input1['Age'], origin='h')
plt.title('Age Distribution')
print(f"Median Age:\t{_input1['Age'].median()}")
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median(), inplace=False)
sns.violinplot(data=_input1, orient='h')
Spendings = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in Spendings:
    _input1[col] = _input1[col].fillna(_input1[col].median(), inplace=False)
    _input0[col] = _input0[col].fillna(_input0[col].median(), inplace=False)
_input1.head(5)
print(f"{_input1['HomePlanet'].unique()}\n{_input1['Destination'].unique()}")
diff_col = ['HomePlanet', 'Destination']
_input1 = pd.concat([_input1, pd.get_dummies(_input1[diff_col])], axis=1)
_input0 = pd.concat([_input0, pd.get_dummies(_input0[diff_col])], axis=1)
_input1 = _input1.drop(diff_col, axis=1, inplace=False)
_input0 = _input0.drop(diff_col, axis=1, inplace=False)
_input1.info()
col = ['VIP', 'CryoSleep']
_input1[col] = _input1[col].astype('int')
_input0[col] = _input0[col].astype('int')
_input1.info()
train_Y = _input1['Transported']
train_X = _input1.drop(columns=['Transported'], axis=1)
print(train_X.shape, train_Y.shape)
(X, X_val, Y, Y_val) = train_test_split(train_X, train_Y, train_size=0.8, random_state=42)
print(X.shape, X_val.shape, Y.shape, Y_val.shape)
RF = RandomForestClassifier(max_depth=9.5)