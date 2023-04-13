import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from catboost import CatBoostClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_val_score, train_test_split
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
(_input1.shape, _input0.shape)
_input1.head(2)
_input0.head(2)
_input1['Transported'].isna().sum()
data = pd.concat([_input1, _input0], axis=0, ignore_index=True)
data = pd.concat([data, data['Cabin'].str.split('/', expand=True)], axis=1)
data = data.rename(columns={0: 'Deck', 1: 'Num', 2: 'Side'}, inplace=False)
data = data.drop(columns=['PassengerId', 'Name', 'Cabin', 'Num'], inplace=False)
data.isna().sum() / data.shape[0]
data.info()
for col in data.columns:
    if data[col].dtype == object:
        print(col, '\x08:')
        print(data[col].value_counts(dropna=False))
data['HomePlanet'] = data['HomePlanet'].fillna('Earth', inplace=False)
data['CryoSleep'] = data['CryoSleep'].fillna(False, inplace=False)
data['Destination'] = data['Destination'].fillna('TRAPPIST-1e', inplace=False)
data['VIP'] = data['VIP'].fillna(False, inplace=False)
data['Deck'] = data['Deck'].fillna('F', inplace=False)
data['Side'] = data['Side'].fillna(method='ffill', inplace=False)
data['Transported'] = data['Transported'].fillna(method='ffill', inplace=False)
for col in data.columns:
    if data[col].dtype != object:
        data[col] = data[col].fillna(data[col].median(), inplace=False)
data.isna().sum()[data.isna().sum() > 0]
for col in data.columns:
    if data[col].dtype == 'float64':
        print(col, '\x08:', data[col].var())
data1 = data.copy()
for col in data1.columns:
    if data1[col].dtype == 'float64':
        data1[col] = (data1[col] - data1[col].mean()) / data1[col].std()
        print(data1[col].var())
data1['Transported'].astype(int)
data1 = pd.get_dummies(data1)
data1.head()
train1 = data1.iloc[:_input1.shape[0], :]
test1 = data1.iloc[_input1.shape[0]:, :]
(train1.shape, test1.shape)
train1.duplicated().sum()
train1 = train1.drop_duplicates(inplace=False)
train1.duplicated().sum()
clf = LocalOutlierFactor(contamination=0.01)
outliers = clf.fit_predict(train1)
train2 = train1[np.where(outliers == 1, True, False)]
train2.head()
train2 = pd.concat([train2, train2], axis=0, ignore_index=True)
x = train2.drop(columns='Transported')
pred = test1.drop(columns='Transported')
y = train2['Transported']
(x.shape, y.shape, pred.shape)
(xtrain, xtest, ytrain, ytest) = train_test_split(x, y, test_size=0.5)
(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
cat = CatBoostClassifier(eval_metric='Accuracy')