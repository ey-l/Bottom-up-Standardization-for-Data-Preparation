import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input1.head()
_input1.tail()
_input1.describe().T
_input1 = _input1.drop('Name', axis=1)
_input1.isnull().sum()
_input1.dtypes
import seaborn as sns
pp = _input1.corr()
sns.heatmap(pp)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
_input1['Transported'] = le.fit_transform(_input1['Transported'])
_input1['HomePlanet'] = le.fit_transform(_input1['HomePlanet'])
_input1['Cabin'] = le.fit_transform(_input1['Cabin'])
_input1['Destination'] = le.fit_transform(_input1['Destination'])
_input1['CryoSleep'] = le.fit_transform(_input1['CryoSleep'])
_input1['VIP'] = le.fit_transform(_input1['VIP'])
import warnings
warnings.filterwarnings('ignore')
count = 1
plt.subplots(figsize=(20, 15))
for i in _input1.columns:
    if _input1[i].dtypes != 'object':
        plt.subplot(6, 7, count)
        sns.distplot(_input1[i])
        count += 1
for i in _input1.columns:
    if _input1[i].dtype == 'object':
        _input1[i] = _input1[i].fillna(_input1[i].mode()[0], inplace=False)
    else:
        _input1[i] = _input1[i].fillna(_input1[i].median(), inplace=False)
print(_input1.isnull().sum())
y = _input1['Transported']
x = _input1.drop('Transported', axis=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
model = RandomForestClassifier(max_depth=10, random_state=42)
(train_X, test_x, train_y, test_y) = train_test_split(x, y, test_size=0.25, random_state=42)