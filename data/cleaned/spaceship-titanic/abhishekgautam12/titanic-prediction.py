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
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(train_data.shape)
train_data.head()
sns.countplot(x=train_data['Transported'], label='count')
sns.countplot(x=train_data.Age, label='count')
sns.countplot(x=train_data.HomePlanet, label='count')
sns.countplot(x=train_data.CryoSleep, label='count')
sns.countplot(x=train_data.VIP, label='count')
train_data.nunique()
train_data.isnull().sum()
train_data.dropna(how='any', axis=0, inplace=True)
train_data.isnull().sum()
train_data[['CryoSleep', 'VIP', 'Transported']] = train_data[['CryoSleep', 'VIP', 'Transported']].astype(int)
train_data.head()
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['HomePlanet', 'Destination'])
encoded_train_data = encoder.fit_transform(train_data)
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