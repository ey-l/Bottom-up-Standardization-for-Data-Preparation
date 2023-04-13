import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head(10)
print('Train shape: ', _input1.shape, 'Test shape:', _input0.shape)
print('Train columns: \n', _input1.columns, '\n\n', 'Test columns: \n', _input0.columns)
print('Train columns: \n', _input1.dtypes, '\n\n', 'Test columns: \n', _input0.dtypes)
print('Train unique values: \n', _input1.nunique(), '\n\n', 'Test unique values: \n', _input0.nunique())
print('Train distinct values: \n', _input1['HomePlanet'].unique(), '\n\n', 'Test distinct values: \n', _input0['HomePlanet'].unique())
print('Train distinct values: \n', _input1['Destination'].unique(), '\n\n', 'Test distinct values: \n', _input0['Destination'].unique())
idsUnique = len(set(_input1.PassengerId))
idsTotal = _input1.shape[0]
idsDupli = idsTotal - idsUnique
print('Train: There are ' + str(idsDupli) + ' duplicate IDs for ' + str(idsTotal) + ' total entries')
idsUnique = len(set(_input0.PassengerId))
idsTotal = _input0.shape[0]
idsDupli = idsTotal - idsUnique
print('Test: There are ' + str(idsDupli) + ' duplicate IDs for ' + str(idsTotal) + ' total entries')
train_missing_val_count = _input1.isnull().sum()
test_missing_val_count = _input0.isnull().sum()
print('Train missing values: \n', train_missing_val_count[train_missing_val_count > 0], '\n\n', 'Test missing values: \n', test_missing_val_count[test_missing_val_count > 0])
_input1 = _input1.fillna(method='ffill')
_input0 = _input0.fillna(method='ffill')
train_missing_val_count = _input1.isnull().sum()
test_missing_val_count = _input0.isnull().sum()
print('Train missing values: \n', train_missing_val_count[train_missing_val_count > 0], '\n\n', 'Test missing values: \n', test_missing_val_count[test_missing_val_count > 0])
_input1['CryoSleep'] = _input1['CryoSleep'].astype(int)
_input1['VIP'] = _input1['VIP'].astype(int)
_input1['Transported'] = _input1['Transported'].astype(int)
_input0['CryoSleep'] = _input0['CryoSleep'].astype(int)
_input0['VIP'] = _input0['VIP'].astype(int)
_input1['HomePlanet'] = _input1['HomePlanet'].replace({'Europa': 0, 'Earth': 1, 'Mars': 2})
_input1['Destination'] = _input1['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2})
_input0['HomePlanet'] = _input0['HomePlanet'].replace({'Europa': 0, 'Earth': 1, 'Mars': 2})
_input0['Destination'] = _input0['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2})
_input1[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = _input1['Cabin'].str.split('/', n=2, expand=True)
print(_input1['Cabin_deck'].unique(), _input1['Cabin_side'].unique())
_input0[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = _input0['Cabin'].str.split('/', n=2, expand=True)
print(_input0['Cabin_deck'].unique(), _input0['Cabin_side'].unique())
_input1['Cabin_deck'] = _input1['Cabin_deck'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})
_input1['Cabin_side'] = _input1['Cabin_side'].replace({'P': 0, 'S': 1})
_input1['Cabin_num'] = _input1['Cabin_num'].astype(int)
_input0['Cabin_deck'] = _input0['Cabin_deck'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7})
_input0['Cabin_side'] = _input0['Cabin_side'].replace({'P': 0, 'S': 1})
_input0['Cabin_num'] = _input0['Cabin_num'].astype(int)
_input1[['gggg', 'pp']] = _input1['PassengerId'].str.split('_', expand=True)
_input1['gggg'] = _input1['gggg'].astype(int)
_input1['pp'] = _input1['pp'].astype(int)
_input0[['gggg', 'pp']] = _input0['PassengerId'].str.split('_', expand=True)
_input0['gggg'] = _input0['gggg'].astype(int)
_input0['pp'] = _input0['pp'].astype(int)
_input1[['First_Name', 'Last_Name']] = _input1['Name'].str.split(' ', expand=True)
_input0[['First_Name', 'Last_Name']] = _input0['Name'].str.split(' ', expand=True)
_input1['First_Name_Initial'] = _input1['First_Name'].str[0]
_input1['First_Name_Initial'] = _input1['First_Name_Initial'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26})
_input0['First_Name_Initial'] = _input0['First_Name'].str[0]
_input0['First_Name_Initial'] = _input0['First_Name_Initial'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26})
_input1.dtypes
_input1['Cabin_deck_x_gggg'] = _input1['Cabin_deck'] * _input1['gggg']
_input0['Cabin_deck_x_gggg'] = _input0['Cabin_deck'] * _input0['gggg']
_input1['Destination_x_CryoSleep'] = _input1['Destination'] * _input1['CryoSleep']
_input0['Destination_x_CryoSleep'] = _input0['Destination'] * _input0['CryoSleep']
corr = _input1.corr()
corr = corr.sort_values(['Transported'], ascending=False, inplace=False)
print(corr.Transported)
categorical_features = _input1.select_dtypes(include=['object']).columns
numerical_features = _input1.select_dtypes(include=['int64', 'float64']).columns
numerical_features = numerical_features.drop('Transported')
print('Numerical features : ', str(len(numerical_features)), numerical_features)
print('Categorical features : ', str(len(categorical_features)), categorical_features)
train_num = _input1[numerical_features]
train_cat = _input1[categorical_features]
test_num = _input0[numerical_features]
test_cat = _input0[categorical_features]
'\nprint("NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))\ntrain_cat = pd.get_dummies(train_cat)\nprint("Remaining NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))\n# Test\nprint("NAs for categorical features in test : " + str(test_cat.isnull().values.sum()))\ntest_cat = pd.get_dummies(test_cat)\nprint("Remaining NAs for categorical features in test : " + str(test_cat.isnull().values.sum()))\n'
_input1.head(11)
_input1.tail(10)
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
y = _input1.Transported
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