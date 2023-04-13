import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_set_eda = _input1.copy()
sns.countplot(x='Transported', data=train_set_eda)
train_set_eda.dtypes
train_set_eda.isnull().sum()
train_set_eda.isnull().sum() / train_set_eda.shape[0]
train_set_eda.describe()
train_set_eda.describe(include=['O'])
(numerical_cols, categorical_cols) = (train_set_eda.dtypes[_input1.dtypes != 'object'].index, train_set_eda.dtypes[_input1.dtypes == 'object'].index)
sns.countplot(data=train_set_eda, x='HomePlanet', hue='Transported')
sns.countplot(data=train_set_eda, x='CryoSleep', hue='Transported')
train_set_eda[['Deck', 'Num', 'Side']] = train_set_eda['Cabin'].str.split('/', expand=True)
train_set_eda = train_set_eda.drop('Cabin', axis=1, inplace=False)
train_set_eda.describe(include=['O'])
sns.countplot(data=train_set_eda, x='Deck', hue='Transported')
sns.countplot(data=train_set_eda, x='Side', hue='Transported')
sns.countplot(data=train_set_eda, x='Destination', hue='Transported')
sns.violinplot(x='Transported', y='Age', data=train_set_eda)
bins = [0, 15, 20, 30, 40, 50, 60, 100]
group_names = ['0-15', '16-20', '21-30', '31-40', '41-50', '51-60', '61-100']
train_set_eda['AgeGroup'] = pd.cut(train_set_eda['Age'], bins, labels=group_names)
sns.countplot(hue='Transported', x='AgeGroup', data=train_set_eda)
sns.countplot(data=train_set_eda, x='VIP', hue='Transported')
cols_to = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
calc_df = train_set_eda[cols_to].copy()
(fig, axes) = plt.subplots(1, 5, figsize=(40, 8))
for (i, col) in enumerate(cols_to[:-1]):
    sns.kdeplot(data=calc_df, x=col, hue='Transported', ax=axes[i], fill=True)
    axes[i].set_title(col)
for col in cols_to[:-1]:
    calc_df[col] = calc_df[col].apply(np.log1p)
(fig, axes) = plt.subplots(1, 5, figsize=(40, 8))
for (i, col) in enumerate(cols_to[:-1]):
    sns.kdeplot(data=calc_df, x=col, hue='Transported', ax=axes[i], fill=True)
    axes[i].set_title(col)
plt.figure(figsize=(8, 6))
corr_matrix = _input1.corr()
sns.heatmap(corr_matrix, annot=True, cmap=sns.diverging_palette(250, 30, l=60, as_cmap=True))
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
import torch.cuda
(X_train, X_test, y_train, y_test) = train_test_split(_input1.drop('Transported', axis=1), _input1['Transported'], test_size=0.2, random_state=42)

class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        df = X.copy()
        df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
        bins = [0, 15, 20, 30, 40, 50, 60, 100]
        group_names = ['0-15', '16-20', '21-30', '31-40', '41-50', '51-60', '61-100']
        df['AgeGroup'] = pd.cut(df['Age'], bins, labels=group_names)
        df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].apply(np.log1p)
        df = df.drop(['PassengerId', 'Name', 'Cabin', 'Num'], axis=1, inplace=False)
        return df
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
column_tranformer = ColumnTransformer(transformers=[('num', numerical_transformer, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']), ('cat', categorical_transformer, ['Side', 'VIP', 'CryoSleep', 'HomePlanet', 'Destination', 'Deck', 'AgeGroup'])])
ml_pipeline = Pipeline(steps=[('preprocessor', Preprocessor()), ('column_transformer', column_tranformer), ('classifier', XGBClassifier())])
parameters = {'classifier__n_estimators': [100, 200, 300], 'classifier__max_depth': [3, 4, 5], 'classifier__learning_rate': [0.1, 0.05, 0.01], 'classifier__objective': ['binary:logistic'], 'classifier__tree_method': ['gpu_hist'] if torch.cuda.is_available() else ['hist']}
grid_search = GridSearchCV(ml_pipeline, parameters, cv=5, n_jobs=-1, verbose=1)