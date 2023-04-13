import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.info()
_input1.isnull().sum()
_input1['VIP'].value_counts()
train1 = _input1.fillna(_input1.mode().iloc[0])
train1.head()
train1['VIP'].value_counts()
_input1['HomePlanet'].value_counts()
train1['HomePlanet'].value_counts()
_input1['CryoSleep'].value_counts()
train1['CryoSleep'].value_counts()
_input1['Destination'].value_counts()
_input1['Transported'].value_counts()
_input1['Age'].mean()
train1['Age'].mean()
_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].mean()
train1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].mean()
import seaborn as sns
import matplotlib.pyplot as plt
train1.hist(bins=50, figsize=(12, 7))
sns.countplot(data=train1, x='HomePlanet', hue='Transported')
sns.countplot(data=train1, x='CryoSleep', hue='Transported')
sns.countplot(data=train1, x='Destination', hue='Transported')
sns.countplot(data=train1, x='VIP', hue='Transported')
plt.figure(figsize=(12, 7))
sns.heatmap(train1.corr(), annot=True, cmap='crest')
y = train1['Transported']
x = train1.drop('Transported', axis=1)
x.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input0.isnull().sum()
test1 = _input0.fillna(_input0.mode().iloc[0])
num_feats = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
num_pipe = Pipeline([('scaler', StandardScaler())])
num_list = list(num_feats)
cat_list = list(cat_feats)
final_pipe = ColumnTransformer([('num', num_pipe, num_list), ('cat', OneHotEncoder(), cat_list)])
x_train = final_pipe.fit_transform(x)
x_train
x_test = final_pipe.transform(test1)
x_test
x.isnull().sum()
test1.isnull().sum()
len(x_train)
len(x_test)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag = BaggingClassifier(DecisionTreeClassifier(class_weight='balanced'), max_samples=0.5, bootstrap=False, n_estimators=300, max_features=0.5)