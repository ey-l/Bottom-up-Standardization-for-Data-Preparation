import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['lines.linewidth'] = 3
plt.style.use('ggplot')
import sklearn
print('Scikit-learn version: ', sklearn.__version__)
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Train samples: ', _input1.shape[0])
print('Test samples: ', _input0.shape[0])
print('Number of features: ', _input0.shape[1])
_input1.head()
_input0.head()
_input1.info()
_input1.describe()
_input1.isna().sum().sort_values(ascending=False)
sns.heatmap(_input1.isna().T, cbar=False, cmap='YlGnBu')
numeric_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
col_expenses = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin']
target = ['Transported']
df_eda = _input1.copy()
ax = sns.countplot(x='Transported', data=df_eda)
ax.set_title('Target distribution', fontsize=16, loc='left')
ax.set_ylabel('Number of observations')
ax = sns.boxplot(x='variable', y='value', data=df_eda[col_expenses].melt())
ax.set_yscale('log')
ax.set_xlabel('')
ax.set_ylabel('Amount')
ax.set_title('Expenses in amenities', fontsize=16, loc='left')
ax = sns.heatmap(data=df_eda[numeric_features + target].corr(), center=0, cmap='vlag', annot=True, linewidths=2)
ax.set_title('Numerical features correlation', fontsize=16, loc='left')
df_eda['SpentSomething'] = df_eda[col_expenses].sum(axis=1) != 0
df_eda.head()
sns.countplot(x='SpentSomething', hue='Transported', data=df_eda)
(fig, ax) = plt.subplots(len(col_expenses), 2, figsize=(10, 16))
for (i, col) in enumerate(col_expenses):
    sns.histplot(df_eda[col], bins=20, ax=ax[i, 0])
    ax[i, 0].set_ylim([0, 500])
    ax[i, 0].set_title(f'{col} (original)')
    ax[i, 0].set_ylabel('')
    ax[i, 0].set_xlabel('')
    sns.histplot(np.log1p(df_eda[col]), bins=20, ax=ax[i, 1])
    ax[i, 1].set_ylim([0, 500])
    ax[i, 1].set_title(f'{col} (log-transform)')
    ax[i, 1].set_ylabel('')
    ax[i, 1].set_xlabel('')
fig.tight_layout()
ax = sns.histplot(x='Age', hue='Transported', data=df_eda, binwidth=2)
ax.set_title('Age distribution', fontsize=16, loc='left')
ax.set_ylabel('')
for col in df_eda.select_dtypes(exclude='number'):
    print(f'{df_eda[col].nunique()} unique values for "{col}" variable:')
    print(df_eda[col].unique())
    print('\n')
(fig, axes) = plt.subplots(2, 2, figsize=(12, 8))
for (col, ax) in zip(['HomePlanet', 'CryoSleep', 'Destination', 'VIP'], axes.flatten()):
    sns.countplot(x=col, hue='Transported', data=df_eda, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title(f'{col} vs. target variable', fontsize=16, loc='left')
fig.tight_layout()
df_eda['Cabin'] = df_eda['Cabin'].str.split('/')
df_eda['Deck'] = df_eda['Cabin'].apply(lambda x: x[0] if type(x) == list else np.nan)
df_eda['CabinNum'] = df_eda['Cabin'].apply(lambda x: x[1] if type(x) == list else np.nan).astype('float64')
df_eda['Side'] = df_eda['Cabin'].apply(lambda x: x[2] if type(x) == list else np.nan)
df_eda['Group'] = df_eda['PassengerId'].apply(lambda x: x.split('_')[0]).astype(np.int64)
df_eda['Alone'] = df_eda['Group'].value_counts(sort=False)[df_eda['Group']].values == 1
df_eda = df_eda.drop('Group', axis=1, inplace=False)
df_eda.head()
(fig, axes) = plt.subplots(1, 3, figsize=(20, 5))
for (col, ax) in zip(['Deck', 'Side', 'Alone'], axes):
    sns.countplot(x=col, hue='Transported', data=df_eda, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title(f'{col} vs. target variable', fontsize=16, loc='left')
ax = sns.histplot(x='CabinNum', data=df_eda, hue='Transported', binwidth=50)
ax.set_title('Cabin number vs. target variable', fontsize=16, loc='left')
ax.set_xlabel('Cabin number')
ax.set_ylabel('')
X = _input1.drop('Transported', axis=1)
y = _input1['Transported']
from sklearn.base import BaseEstimator, TransformerMixin

class CabinTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, addCabinDeck=True, addCabinNum=True, addCabinSide=True):
        self.addCabinDeck = addCabinDeck
        self.addCabinNum = addCabinNum
        self.addCabinSide = addCabinSide

    def fit(self, X, y=None):
        self.cols = X.columns
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        cabin_list = X_['Cabin'].str.split('/')
        if self.addCabinDeck:
            X_['CabinDeck'] = cabin_list.apply(lambda x: x[0] if type(x) == list else np.nan).astype('category')
        if self.addCabinNum:
            X_['CabinNum'] = cabin_list.apply(lambda x: x[1] if type(x) == list else np.nan).astype('float64')
        if self.addCabinSide:
            X_['CabinSide'] = cabin_list.apply(lambda x: x[2] if type(x) == list else np.nan).astype('category')
        return X_

    def get_feature_names_out(self):
        return self.cols
X_trans = CabinTransformer(addCabinNum=False).fit_transform(X)
X_trans.head()

class FeatureAdderTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, addSpentSomething=True, addAlone=True):
        self.addSpentSomething = addSpentSomething
        self.addAlone = addAlone
        self.columns = col_expenses

    def fit(self, X, y=None):
        self.cols = X.columns
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        if self.addSpentSomething:
            X_['SpentSomething'] = X_[self.columns].sum(axis=1) != 0
        if self.addAlone:
            group = X_['PassengerId'].apply(lambda x: x.split('_')[0]).astype(np.int64)
            X_['Alone'] = group.value_counts(sort=False)[group].values == 1
        X_[['HomePlanet', 'Destination']] = X_[['HomePlanet', 'Destination']].astype('category')
        X_[['CryoSleep', 'VIP']] = X_[['HomePlanet', 'Destination']].astype('bool')
        return X_

    def get_feature_names_out(self):
        return self.cols
X_trans = FeatureAdderTransformer(addSpentSomething=True, addAlone=True).fit_transform(X)
X_trans.head()
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
expenditures_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('log', FunctionTransformer(np.log1p, validate=True)), ('scaler', StandardScaler())])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))])
ct = ColumnTransformer(transformers=[('drop', 'drop', ['Cabin', 'PassengerId', 'Name']), ('exp', expenditures_transformer, col_expenses), ('cat', categorical_transformer, make_column_selector(dtype_include=['category', 'bool'])), ('num', numeric_transformer, make_column_selector(pattern='Age|CabinNum', dtype_include='number'))], remainder='passthrough', verbose=False)
preprocessor = Pipeline([('add_feat', FeatureAdderTransformer()), ('cabin_trans', CabinTransformer()), ('column_trans', ct)])
from sklearn import set_config
set_config(display='diagram')