import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, RocCurveDisplay, roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(df.shape)
df.head()
df.drop(['PassengerId', 'Name'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name'], axis=1, inplace=True)
df[['Cabin1', 'Cabin2', 'Cabin3']] = df['Cabin'].str.split('/', expand=True)
df_test[['Cabin1', 'Cabin2', 'Cabin3']] = df_test['Cabin'].str.split('/', expand=True)
df.drop('Cabin', axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)
df.info()
for column in df.select_dtypes(include=object).columns:
    df[column].fillna(df[column].mode()[0], inplace=True)
for column in df_test.select_dtypes(include=object).columns:
    df_test[column].fillna(df_test[column].mode()[0], inplace=True)
for column in df.select_dtypes(exclude=object).columns:
    df[column].fillna(df[column].mean(), inplace=True)
for column in df_test.select_dtypes(exclude=object).columns:
    df_test[column].fillna(df_test[column].mean(), inplace=True)
encoder = LabelEncoder()
mappings = {}
for column in df.columns:
    if len(df[column].unique()) == 2:
        df[column] = encoder.fit_transform(df[column])
        if column != 'Transported':
            df_test[column] = encoder.transform(df_test[column])
        encoder_mappings = {index: label for (index, label) in enumerate(encoder.classes_)}
        mappings[column] = encoder_mappings
df_test.head()
mappings
df['Cabin2'].unique()
df['Cabin2'] = df['Cabin2'].astype(int)
df_test['Cabin2'] = df_test['Cabin2'].astype(int)
df_ = df.copy()
for column in df_.select_dtypes(include=object).columns:
    dummies = pd.get_dummies(df[column])
    df = pd.concat((df, dummies), axis=1)
    df.drop(column, axis=1, inplace=True)
df_ = df_test.copy()
for column in df_.select_dtypes(include=object).columns:
    dummies = pd.get_dummies(df_test[column])
    df_test = pd.concat((df_test, dummies), axis=1)
    df_test.drop(column, axis=1, inplace=True)
X = df.drop('Transported', axis=1)
y = df['Transported']
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(df_test)
svc = SVC(probability=True)