import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(test_df)
test_df.describe()
(test_df['Age'] == 0).sum()
test_df['Age'] = test_df['Age'].replace(0, np.nan)
(test_df['Age'] == 0).sum()
test_df.isnull().sum()
test_df.isnull().sum() / len(test_df)
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(train_df)
train_df.describe()
(train_df['Age'] == 0).sum()
train_df['Age'] = train_df['Age'].replace(0, np.nan)
(train_df['Age'] == 0).sum()
train_df.isnull().sum()
train_df.isnull().sum() / len(train_df)
train_df = train_df.drop(['Name'], axis=1)
train_df.head()
train_df.info()
test_df = test_df.drop(['Name'], axis=1)
test_df.head()
test_df.info()
train_df[['Deck', 'DeckNum', 'Side']] = train_df.Cabin.str.split('/', expand=True)
test_df[['Deck', 'DeckNum', 'Side']] = test_df.Cabin.str.split('/', expand=True)
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)
train_df.head()
test_df.head()
y = train_df.Transported
x_train_dataset = train_df.drop(['PassengerId', 'Transported'], axis=1)
x_test_dataset = test_df.drop(['PassengerId'], axis=1)
print(x_train_dataset.shape)
print(y.shape)
print(x_test_dataset.shape)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
x_train_dataset = x_train_dataset.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
imputer = KNNImputer(n_neighbors=5)
x_train_tra = imputer.fit_transform(x_train_dataset)
x_train_tra = pd.DataFrame(x_train_tra)
print(x_train_tra)
x_train_tra.columns = x_train_dataset.columns.values
x_train_tra.head()
x_train_tra.isnull().sum()
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
x_test_dataset = x_test_dataset.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
imputer = KNNImputer(n_neighbors=3)
x_test_tra = imputer.fit_transform(x_test_dataset)
x_test_tra = pd.DataFrame(x_test_tra)
print(x_test_tra)
x_test_tra.isnull().sum()
x_test_tra.columns = x_test_dataset.columns.values
x_test_tra.head()
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
x_train_tra = scaler.fit_transform(x_train_tra)
x_test_tra = scaler.fit_transform(x_test_tra)
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
cv = KFold(n_splits=5, random_state=0, shuffle=True)
model = LogisticRegression()
scores = cross_val_score(model, x_train_tra, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
from sklearn.linear_model import LogisticRegressionCV