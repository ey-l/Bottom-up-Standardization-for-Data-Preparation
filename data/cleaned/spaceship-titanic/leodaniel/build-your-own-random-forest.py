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
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(df.shape)
df.head()
count = 1
for c in df.columns:
    print(f'{count} - {c}')
    print(f'- # of unique elements: {df[c].nunique()}')
    print(f'- Sample: {df[c].unique()[0:20]}')
    print(f'- Dtype: {df[c].dtype}')
    print(f'- # of missing values: {df[c].isnull().sum()} of {df.shape[0]}')
    print(f'- % of missing values: {np.round(df[c].isnull().sum() / df.shape[0], 3)}')
    if df[c].dtype == int or df[c].dtype == float:
        s = '- Statistics:\n'
        me = np.round(df[c].mean(), 2)
        st = np.round(df[c].std(), 2)
        s += f'-- Mean (std): {me} ({st})\n'
        q1 = np.round(df[c].quantile(0.25), 2)
        q2 = np.round(df[c].quantile(0.5), 2)
        q3 = np.round(df[c].quantile(0.75), 2)
        s += f'-- Quantiles: q1={q1}, q2={q2}, q3={q3}\n'
        s += f'-- Min {df[c].min()}\n'
        s += f'-- Max {df[c].max()}'
        print(s)
    print('=' * 30)
    count += 1
df.drop(columns=['PassengerId', 'Cabin', 'Name'], inplace=True)
print(df.shape)
df.head()
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
numeric_transformer = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())])
Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_features = ['HomePlanet', 'Destination']
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
binary_features = ['CryoSleep', 'VIP']
binary_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OrdinalEncoder())])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features), ('bin', binary_transformer, binary_features)])
X = df.drop(columns=['Transported'])
y = df['Transported']
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