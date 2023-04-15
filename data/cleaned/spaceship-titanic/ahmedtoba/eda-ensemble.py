import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv').drop(['PassengerId', 'Name'], axis=1)
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv').drop(['PassengerId', 'Name'], axis=1)
sample = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train_data.head(15)
train_data.info()
test_data.head()
test_data.describe().T
train_data.isna().sum().sort_values(ascending=False)
test_data.isna().sum().sort_values(ascending=False)
obj_train_features = []
for col in train_data.columns:
    if np.dtype(train_data[col]) == 'object':
        obj_train_features.append(col)
print(len(obj_train_features))
obj_test_features = []
for col in test_data.columns:
    if np.dtype(test_data[col]) == 'object':
        obj_test_features.append(col)
print(len(obj_test_features))
num_train_features = []
for col in train_data.columns:
    if np.dtype(train_data[col]) == 'float64':
        num_train_features.append(col)
print(len(num_train_features))
list(obj_train_features)
list(obj_test_features)
train_data['HomePlanet'].value_counts()
test_data['HomePlanet'].value_counts()
train_data['Destination'].value_counts()
test_data['Destination'].value_counts()
train_data['VIP'].value_counts()
train_data['CryoSleep'].value_counts()

def clean(data):
    data['HomePlanet'] = data['HomePlanet'].fillna('Earth')
    data['HomePlanet'] = data['HomePlanet'].map({'Europa': 0, 'Earth': 1, 'Mars': 2})
    data['Destination'] = data['Destination'].fillna('TRAPPIST-1e')
    data['Destination'] = data['Destination'].map({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2})
    data['VIP'] = data['VIP'].fillna(False).astype(int)
    data['CryoSleep'] = data['CryoSleep'].fillna(False).astype(int)
    return data
train_data = clean(train_data)
test_data = clean(test_data)
list(num_train_features)

def full_nuls(data):
    for col in num_train_features:
        data[col] = data[col].fillna(data[col].mean())
    return data
train_data = full_nuls(train_data)
test_data = full_nuls(test_data)
train_data.isna().sum().sort_values(ascending=False)

def D_cabin(Cabin):
    try:
        return Cabin.split('/')[0]
    except:
        return np.NaN

def S_cabin(Cabin):
    try:
        return Cabin.split('/')[2]
    except:
        return np.NaN
train_data['Dec_cabin'] = train_data['Cabin'].apply(lambda x: D_cabin(x))
train_data['Sid_cabin'] = train_data['Cabin'].apply(lambda x: S_cabin(x))
train_data = train_data.drop(['Cabin'], axis=1)
test_data['Dec_cabin'] = test_data['Cabin'].apply(lambda x: D_cabin(x))
test_data['Sid_cabin'] = test_data['Cabin'].apply(lambda x: S_cabin(x))
test_data = test_data.drop(['Cabin'], axis=1)
train_data['Dec_cabin'].value_counts(dropna=False)
train_data['Dec_cabin'].iloc[:2500] = train_data['Dec_cabin'].iloc[:2500].fillna('f', inplace=True)
train_data['Dec_cabin'].iloc[2500:] = train_data['Dec_cabin'].iloc[2500:].fillna('G', inplace=True)
train_data['Sid_cabin'].value_counts(dropna=False)
np.random.seed(43)
data = np.random.choice(a=list(train_data['Sid_cabin'].value_counts().index), p=[0.5, 0.5], size=199)
fill = pd.DataFrame(index=train_data.index[train_data['Sid_cabin'].isnull()], data=data, columns=['Sid_cabin'])
train_data.fillna(fill, inplace=True)
test_data['Sid_cabin'].value_counts(dropna=False)
np.random.seed(43)
data = np.random.choice(a=list(test_data['Sid_cabin'].value_counts().index), p=[0.5, 0.5], size=100)
fill = pd.DataFrame(index=test_data.index[test_data['Sid_cabin'].isnull()], data=data, columns=['Sid_cabin'])
test_data.fillna(fill, inplace=True)
train_data['Sid_cabin'] = train_data['Sid_cabin'].map({'S': 1, 'P': 0})
test_data['Sid_cabin'] = test_data['Sid_cabin'].map({'S': 1, 'P': 0})
label_model = LabelEncoder()
train_data['Dec_cabin'] = label_model.fit_transform(train_data['Dec_cabin'])
test_data['Dec_cabin'] = label_model.fit_transform(test_data['Dec_cabin'])
train_data.head()
y = train_data['Transported']
train_data = train_data.drop(['Transported'], axis=1)
X = train_data
X_test = test_data
SKF_model = StratifiedKFold(n_splits=5)
XGB_model = XGBClassifier(n_estimators=100, learning_rate=0.07, booster='gbtree', gamma=0.5, reg_alpha=0.5, reg_lambda=0.5, base_score=0.2)
RF_model = RandomForestClassifier(n_estimators=120, criterion='gini', min_samples_split=1.0, min_samples_leaf=0.5, max_leaf_nodes=4)
VC_model = VotingClassifier(estimators=[('XGB', XGB_model), ('Rf', RF_model)], voting='hard')
test_list = []
for (count, (train_idx, test_idx)) in enumerate(SKF_model.split(X, y)):
    X_train = X.iloc[train_idx]
    X_valid = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[test_idx]
    print('*************fold(', count + 1, ')***************')