import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
file_path = '_data/input/house-prices-advanced-regression-techniques'
train_data = os.path.join(file_path, 'train.csv')
test_data = os.path.join(file_path, 'test.csv')
df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data)
print([x for x in df_train.columns if x not in df_test.columns])
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class Dropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        retval = X.drop(columns=['Id']).select_dtypes(exclude=['object'])
        (retval['Neighborhood'], retval['GarageType']) = [X['Neighborhood'], X['GarageType']]
        return retval
pipe = Pipeline([('dropper', Dropper())])
y_train = df_train.pop('SalePrice')
X_train = pipe.fit_transform(df_train)
X_test = pipe.fit_transform(df_test)
X_train[['GarageType']].isna().sum() * 100 / len(X_train)
import seaborn as sns
sns.set_theme(style='darkgrid')
sns.catplot(data=X_train, x='GarageType', kind='count', palette='pastel')

class Imputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['GarageType'] = X['GarageType'].fillna('Attchd')
        return X
pipe = Pipeline([('dropper', Dropper()), ('imputer', Imputer())])
X_train = pipe.fit_transform(df_train)
X_test = pipe.fit_transform(df_test)
from sklearn.preprocessing import OneHotEncoder

class FeatureEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Neighborhood']]).toarray()
        for (idx, col) in enumerate(encoder.categories_[0][:-1]):
            X[col] = matrix[:, idx]
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['GarageType']]).toarray()
        for (idx, col) in enumerate(encoder.categories_[0][:-1]):
            X[col] = matrix[:, idx]
        return X.drop(columns=['Neighborhood', 'GarageType'])
pipe = Pipeline([('dropper', Dropper()), ('imputer', Imputer()), ('feature_encoder', FeatureEncoder())])
X_train = pipe.fit_transform(df_train)
X_test = pipe.fit_transform(df_test)
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
params = {'max_depth': [5, 10], 'learning_rate': [0.01, 0.02], 'n_estimators': [100, 500]}
model = XGBRegressor(seed=42)
grid = GridSearchCV(estimator=model, param_grid=params, cv=5)