import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(_input1.shape)
_input1.head()
sns.countplot(x=_input1['Transported'], label='count')
sns.countplot(x=_input1.Age, label='count')
sns.countplot(x=_input1.HomePlanet, label='count')
sns.countplot(x=_input1.CryoSleep, label='count')
sns.countplot(x=_input1.VIP, label='count')
_input1.nunique()
_input1.isnull().sum()
_input1 = _input1.dropna(how='any', axis=0, inplace=False)
_input1.isnull().sum()
_input1[['CryoSleep', 'VIP', 'Transported']] = _input1[['CryoSleep', 'VIP', 'Transported']].astype(int)
_input1.head()
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['HomePlanet', 'Destination'])
encoded_train_data = encoder.fit_transform(_input1)
encoded_train_data.head()
sns.heatmap(encoded_train_data.corr())
feature_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Age']
dataset = encoded_train_data[feature_cols]
target = encoded_train_data['Transported']
print(dataset.dtypes)
dataset.head()
target.head()
(X_train, X_test, y_train, y_test) = train_test_split(dataset, target)
rfc = RandomForestClassifier(n_estimators=100)