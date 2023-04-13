import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from mlxtend.feature_selection import ColumnSelector
from yellowbrick.model_selection import ValidationCurve
SEED = 123
np.random.seed(SEED)
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(_input1.shape)
_input1.head()
count = 1
for c in _input1.columns:
    print(f'{count} - {c}')
    print(f'- # of unique elements: {_input1[c].nunique()}')
    print(f'- Sample: {_input1[c].unique()[0:20]}')
    print(f'- Dtype: {_input1[c].dtype}')
    print(f'- # of missing values: {_input1[c].isnull().sum()} of {_input1.shape[0]}')
    print(f'- % of missing values: {np.round(_input1[c].isnull().sum() / _input1.shape[0], 3)}')
    if _input1[c].dtype == int or _input1[c].dtype == float:
        s = '- Statistics:\n'
        me = np.round(_input1[c].mean(), 2)
        st = np.round(_input1[c].std(), 2)
        s += f'-- Mean (std): {me} ({st})\n'
        q1 = np.round(_input1[c].quantile(0.25), 2)
        q2 = np.round(_input1[c].quantile(0.5), 2)
        q3 = np.round(_input1[c].quantile(0.75), 2)
        s += f'-- Quantiles: q1={q1}, q2={q2}, q3={q3}\n'
        s += f'-- Min {_input1[c].min()}\n'
        s += f'-- Max {_input1[c].max()}'
        print(s)
    print('=' * 30)
    count += 1
_input1 = _input1.drop(columns=['PassengerId', 'Cabin', 'Name'], inplace=False)
print(_input1.shape)
_input1.head()
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
numeric_transformer = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())])
Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_features = ['HomePlanet', 'Destination']
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
binary_features = ['CryoSleep', 'VIP']
binary_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OrdinalEncoder())])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features), ('bin', binary_transformer, binary_features)])
X = _input1.drop(columns=['Transported'])
y = _input1['Transported']
y = LabelEncoder().fit_transform(y)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print('-' * 20)
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')
acc_dt = {'max_depth': [], 'acc': [], 'train_test': []}
max_depth = np.arange(2, 21, 1)
for md in tqdm(max_depth):
    pipe_dt = Pipeline([('preprocessor', preprocessor), ('estimator', DecisionTreeClassifier(random_state=SEED, max_depth=md))])