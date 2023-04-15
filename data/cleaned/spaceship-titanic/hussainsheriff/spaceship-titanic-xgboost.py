import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.head(10)
print('Train shape: ', train.shape, 'Test shape:', test.shape)
print('Train columns: \n', train.columns, '\n\n', 'Test columns: \n', test.columns)
print('Train columns: \n', train.dtypes, '\n\n', 'Test columns: \n', test.dtypes)
print('Train unique values: \n', train.nunique(), '\n\n', 'Test unique values: \n', test.nunique())
print('Train distinct values: \n', train['HomePlanet'].unique(), '\n\n', 'Test distinct values: \n', test['HomePlanet'].unique())
print('Train distinct values: \n', train['Destination'].unique(), '\n\n', 'Test distinct values: \n', test['Destination'].unique())
idsUnique = len(set(train.PassengerId))
idsTotal = train.shape[0]
idsDupli = idsTotal - idsUnique
print('Train: There are ' + str(idsDupli) + ' duplicate IDs for ' + str(idsTotal) + ' total entries')
idsUnique = len(set(test.PassengerId))
idsTotal = test.shape[0]
idsDupli = idsTotal - idsUnique
print('Test: There are ' + str(idsDupli) + ' duplicate IDs for ' + str(idsTotal) + ' total entries')
train_missing_val_count = train.isnull().sum()
test_missing_val_count = test.isnull().sum()
print('Train missing values: \n', train_missing_val_count[train_missing_val_count > 0], '\n\n', 'Test missing values: \n', test_missing_val_count[test_missing_val_count > 0])
train = train.fillna(method='ffill')
test = test.fillna(method='ffill')
train_missing_val_count = train.isnull().sum()
test_missing_val_count = test.isnull().sum()
print('Train missing values: \n', train_missing_val_count[train_missing_val_count > 0], '\n\n', 'Test missing values: \n', test_missing_val_count[test_missing_val_count > 0])
train['CryoSleep'] = train['CryoSleep'].astype(int)
train['VIP'] = train['VIP'].astype(int)
train['Transported'] = train['Transported'].astype(int)
test['CryoSleep'] = test['CryoSleep'].astype(int)
test['VIP'] = test['VIP'].astype(int)
train['HomePlanet'] = train['HomePlanet'].replace({'Europa': 0, 'Earth': 1, 'Mars': 2})
train['Destination'] = train['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2})
test['HomePlanet'] = test['HomePlanet'].replace({'Europa': 0, 'Earth': 1, 'Mars': 2})
test['Destination'] = test['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2})
train[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = train['Cabin'].str.split('/', n=2, expand=True)
print(train['Cabin_deck'].unique(), train['Cabin_side'].unique())
test[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = test['Cabin'].str.split('/', n=2, expand=True)
print(test['Cabin_deck'].unique(), test['Cabin_side'].unique())
train['Cabin_deck'] = train['Cabin_deck'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})
train['Cabin_side'] = train['Cabin_side'].replace({'P': 0, 'S': 1})
train['Cabin_num'] = train['Cabin_num'].astype(int)
test['Cabin_deck'] = test['Cabin_deck'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7})
test['Cabin_side'] = test['Cabin_side'].replace({'P': 0, 'S': 1})
test['Cabin_num'] = test['Cabin_num'].astype(int)
train[['gggg', 'pp']] = train['PassengerId'].str.split('_', expand=True)
train['gggg'] = train['gggg'].astype(int)
train['pp'] = train['pp'].astype(int)
test[['gggg', 'pp']] = test['PassengerId'].str.split('_', expand=True)
test['gggg'] = test['gggg'].astype(int)
test['pp'] = test['pp'].astype(int)
train[['First_Name', 'Last_Name']] = train['Name'].str.split(' ', expand=True)
test[['First_Name', 'Last_Name']] = test['Name'].str.split(' ', expand=True)
train['First_Name_Initial'] = train['First_Name'].str[0]
train['First_Name_Initial'] = train['First_Name_Initial'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26})
test['First_Name_Initial'] = test['First_Name'].str[0]
test['First_Name_Initial'] = test['First_Name_Initial'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26})
train.dtypes
train['Cabin_deck_x_gggg'] = train['Cabin_deck'] * train['gggg']
test['Cabin_deck_x_gggg'] = test['Cabin_deck'] * test['gggg']
train['Destination_x_CryoSleep'] = train['Destination'] * train['CryoSleep']
test['Destination_x_CryoSleep'] = test['Destination'] * test['CryoSleep']
corr = train.corr()
corr.sort_values(['Transported'], ascending=False, inplace=True)
print(corr.Transported)
categorical_features = train.select_dtypes(include=['object']).columns
numerical_features = train.select_dtypes(include=['int64', 'float64']).columns
numerical_features = numerical_features.drop('Transported')
print('Numerical features : ', str(len(numerical_features)), numerical_features)
print('Categorical features : ', str(len(categorical_features)), categorical_features)
train_num = train[numerical_features]
train_cat = train[categorical_features]
test_num = test[numerical_features]
test_cat = test[categorical_features]
'\nprint("NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))\ntrain_cat = pd.get_dummies(train_cat)\nprint("Remaining NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))\n# Test\nprint("NAs for categorical features in test : " + str(test_cat.isnull().values.sum()))\ntest_cat = pd.get_dummies(test_cat)\nprint("Remaining NAs for categorical features in test : " + str(test_cat.isnull().values.sum()))\n'
train.head(11)
train.tail(10)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval
pd.set_option('display.float_format', lambda x: '%.3f' % x)

X = train_num
print('New number of features : ' + str(X.shape[1]))
y = train.Transported
print(y)
(X_train, X_validation, y_train, y_validation) = train_test_split(X, y, test_size=0.3, random_state=0)
print('X_train : ' + str(X_train.shape))
print('X_validation : ' + str(X_validation.shape))
print('y_train : ' + str(y_train.shape))
print('y_validation : ' + str(y_validation.shape))
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_validation = pd.DataFrame(sc.transform(X_validation), index=X_validation.index, columns=X_validation.columns)
X_train.describe().T
space = {'learning_rate': hp.choice('learning_rate', [0.0001, 0.001, 0.01, 0.1, 1]), 'max_depth': hp.choice('max_depth', range(3, 21, 3)), 'gamma': hp.choice('gamma', [i / 10.0 for i in range(0, 5)]), 'colsample_bytree': hp.choice('colsample_bytree', [i / 10.0 for i in range(3, 10)]), 'reg_alpha': hp.choice('reg_alpha', [1e-05, 0.01, 0.1, 1, 10, 100]), 'reg_lambda': hp.choice('reg_lambda', [1e-05, 0.01, 0.1, 1, 10, 100])}
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

def objective(params):
    xgboost = XGBClassifier(seed=0, **params)
    score = cross_val_score(estimator=xgboost, X=X_train, y=y_train, cv=kfold, scoring='recall', n_jobs=-1).mean()
    loss = -score
    return {'loss': loss, 'params': params, 'status': STATUS_OK}
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=48)
print(best)
print(space_eval(space, best))