import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
sns.set(color_codes=True)
np.random.seed(42)
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Raw data:')
_input1.head()
_input1.hist(figsize=(15, 10))
_input1 = _input1.drop(labels=['PassengerId', 'Cabin', 'Name'], axis=1)
test_passenger_ids = _input0['PassengerId']
_input0 = _input0.drop(labels=['PassengerId', 'Cabin', 'Name'], axis=1)
print(_input0.head())
_input1.head()
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
num_imputer = SimpleImputer(strategy='mean')
_input1[numerical_cols] = pd.DataFrame(num_imputer.fit_transform(_input1[numerical_cols]), columns=numerical_cols)
_input0[numerical_cols] = pd.DataFrame(num_imputer.fit_transform(_input0[numerical_cols]), columns=numerical_cols)
print('NaN value count in each column:')
_input1.isna().sum()
class_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
cls_imputer = SimpleImputer(strategy='most_frequent')
_input1[class_cols] = pd.DataFrame(cls_imputer.fit_transform(_input1[class_cols]), columns=class_cols)
_input0[class_cols] = pd.DataFrame(cls_imputer.fit_transform(_input0[class_cols]), columns=class_cols)
print('NaN value count in each column:')
_input1.isnull().sum()
class_cols.append('Transported')
for col in class_cols:
    le = LabelEncoder()
    _input1[col] = le.fit_transform(_input1[col])
class_cols.remove('Transported')
for col in class_cols:
    le = LabelEncoder()
    _input0[col] = le.fit_transform(_input0[col])
_input1.head()
(fig, ax) = plt.subplots(figsize=(15, 10))
sns.heatmap(_input1.corr(), annot=True, ax=ax)
(X_train, X_valid, y_train, y_valid) = train_test_split(_input1.loc[:, _input1.columns != 'Transported'], _input1[['Transported']], test_size=0.1)
print('Training dataset shapes:', X_train.shape, y_train.shape)
print('Validation dataset shapes:', X_valid.shape, y_valid.shape)
tree = DecisionTreeClassifier(random_state=42)