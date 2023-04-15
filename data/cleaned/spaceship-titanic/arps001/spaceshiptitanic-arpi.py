import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data[['cabin-a', 'cabin-b', 'cabin-c']] = train_data['Cabin'].str.split('/', expand=True)
test_data[['cabin-a', 'cabin-b', 'cabin-c']] = test_data['Cabin'].str.split('/', expand=True)
train_data['cabin-b'] = train_data['cabin-b'].astype('float')
test_data['cabin-b'] = test_data['cabin-b'].astype('float')
train_data['Transported'] = 1 * train_data['Transported']
X = train_data.drop('Transported', axis=1)
y = train_data['Transported']
Xtest = test_data
drop_feats = ['PassengerId', 'Name', 'Cabin']
X = X.drop(drop_feats, axis=1)
Xtest = Xtest.drop(drop_feats, axis=1)
cat_cols = [o for o in X.columns if X[o].dtype == 'object' and (len(X[o].unique()) > 0 and len(X[o].unique()) < 10)]
num_cols = [o for o in X.columns if X[o].dtype == 'float64' or X[o].dtype == 'int64']
print('cats', cat_cols)
print('nums', num_cols)
from sklearn.impute import SimpleImputer
imputer1 = SimpleImputer()
X[num_cols] = pd.DataFrame(imputer1.fit_transform(X[num_cols]))
Xtest[num_cols] = pd.DataFrame(imputer1.transform(Xtest[num_cols]))
imputer2 = SimpleImputer(strategy='most_frequent')
X[cat_cols] = pd.DataFrame(imputer2.fit_transform(X[cat_cols]))
Xtest[cat_cols] = pd.DataFrame(imputer2.transform(Xtest[cat_cols]))
for i in X.columns:
    print(i, '->', sum(X[i].isnull()))
sum(y.isnull())
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
X[cat_cols] = pd.DataFrame(encoder.fit_transform(X[cat_cols]))
Xtest[cat_cols] = pd.DataFrame(encoder.transform(Xtest[cat_cols]))
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.2)
corr_matrix = train_data.corr()
corr_matrix['Transported'].sort_values(ascending=False)
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
scatter_matrix(train_data, figsize=(20, 10))


def crossvalidate(model, X, y, cv, scoring):
    score = -1 * cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(score)
    print(score.mean())
    return score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
model1_fin = RandomForestRegressor(max_features=7, n_estimators=500, random_state=0)