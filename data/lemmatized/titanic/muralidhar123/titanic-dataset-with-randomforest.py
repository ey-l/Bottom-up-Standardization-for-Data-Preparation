import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head()
_input1['Cabin'].isnull().sum()
_input1.isnull().sum()
_input1['Survived'].unique()
_input1['Age'].unique()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean())
_input1['Embarked'].unique()
_input1['Embarked'] = _input1['Embarked'].fillna(_input1['Embarked'].median)
_input1.head()
_input1 = _input1.drop(labels='Cabin', axis=1)
_input0
_input0.isnull().sum()
_input0 = _input0.drop(labels='Cabin', axis=1)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean())
_input0.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
_input1['Name'] = label.fit_transform(_input1['Name'])
_input1['Name']
_input1['Age'] = label.fit_transform(_input1['Age'])
_input1['Sex'] = label.fit_transform(_input1['Sex'])
_input1['SibSp'] = label.fit_transform(_input1['SibSp'])
_input1['Parch'] = label.fit_transform(_input1['Parch'])
_input1['Ticket'] = label.fit_transform(_input1['Ticket'])
_input1['Fare'] = label.fit_transform(_input1['Fare'])
_input1['Embarked'] = label.fit_transform(_input1['Embarked'].astype(str))
_input1.head()
_input0.head()
_input0['Name'] = label.fit_transform(_input0['Name'])
_input0['Age'] = label.fit_transform(_input0['Age'])
_input0['Sex'] = label.fit_transform(_input0['Sex'])
_input0['SibSp'] = label.fit_transform(_input0['SibSp'])
_input0['Parch'] = label.fit_transform(_input0['Parch'])
_input0['Ticket'] = label.fit_transform(_input0['Ticket'])
_input0['Fare'] = label.fit_transform(_input0['Fare'])
_input0['Embarked'] = label.fit_transform(_input0['Embarked'].astype(str))
_input0.head()
x = _input1
target = _input0
x.head()
target.head()
X = x.drop(labels=['PassengerId', 'Survived'], axis=1)
y = x['Survived']
X_scaled = scaler.fit_transform(X)
X_scaled
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y, test_size=0.7, random_state=120)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
xg = xgb.XGBClassifier(random_state=1, learning_rate=0.01)