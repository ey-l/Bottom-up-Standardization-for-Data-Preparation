import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, RocCurveDisplay, roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(_input1.shape)
_input1.head()
_input1 = _input1.drop(['PassengerId', 'Name'], axis=1, inplace=False)
_input0 = _input0.drop(['PassengerId', 'Name'], axis=1, inplace=False)
_input1[['Cabin1', 'Cabin2', 'Cabin3']] = _input1['Cabin'].str.split('/', expand=True)
_input0[['Cabin1', 'Cabin2', 'Cabin3']] = _input0['Cabin'].str.split('/', expand=True)
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
_input1.info()
for column in _input1.select_dtypes(include=object).columns:
    _input1[column] = _input1[column].fillna(_input1[column].mode()[0], inplace=False)
for column in _input0.select_dtypes(include=object).columns:
    _input0[column] = _input0[column].fillna(_input0[column].mode()[0], inplace=False)
for column in _input1.select_dtypes(exclude=object).columns:
    _input1[column] = _input1[column].fillna(_input1[column].mean(), inplace=False)
for column in _input0.select_dtypes(exclude=object).columns:
    _input0[column] = _input0[column].fillna(_input0[column].mean(), inplace=False)
encoder = LabelEncoder()
mappings = {}
for column in _input1.columns:
    if len(_input1[column].unique()) == 2:
        _input1[column] = encoder.fit_transform(_input1[column])
        if column != 'Transported':
            _input0[column] = encoder.transform(_input0[column])
        encoder_mappings = {index: label for (index, label) in enumerate(encoder.classes_)}
        mappings[column] = encoder_mappings
_input0.head()
mappings
_input1['Cabin2'].unique()
_input1['Cabin2'] = _input1['Cabin2'].astype(int)
_input0['Cabin2'] = _input0['Cabin2'].astype(int)
df_ = _input1.copy()
for column in df_.select_dtypes(include=object).columns:
    dummies = pd.get_dummies(_input1[column])
    _input1 = pd.concat((_input1, dummies), axis=1)
    _input1 = _input1.drop(column, axis=1, inplace=False)
df_ = _input0.copy()
for column in df_.select_dtypes(include=object).columns:
    dummies = pd.get_dummies(_input0[column])
    _input0 = pd.concat((_input0, dummies), axis=1)
    _input0 = _input0.drop(column, axis=1, inplace=False)
X = _input1.drop('Transported', axis=1)
y = _input1['Transported']
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(_input0)
svc = SVC(probability=True)