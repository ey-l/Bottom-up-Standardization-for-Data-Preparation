import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
file_path = 'data/input/spaceship-titanic/train.csv'
data = pd.read_csv(file_path, index_col='PassengerId')
print(data.columns)
data.shape
data.head()
data.dtypes
data.isnull().sum()
data.select_dtypes('object').apply(pd.Series.nunique, axis=0)
data = data.drop(['Cabin', 'Name'], axis=1)
X = data.drop(['Transported'], axis=1)
print(X.columns)
y = data.Transported
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, random_state=0)
categorical_cols = [col for col in X_train.columns if data[col].dtype in ['object', 'bool']]
numerical_cols = [col for col in X_train.columns if data[col].dtypes == 'float64']
print('list of categorical columns:', categorical_cols)
print('list of numerical columns:', numerical_cols)
total_cols = categorical_cols + numerical_cols
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
num_impute = SimpleImputer(strategy='mean')
cat_transform = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', num_impute, numerical_cols), ('cat', cat_transform, categorical_cols)])
print(X_train.shape)
print(y_train.shape)
y_train.head()
from sklearn import svm
from sklearn.metrics import accuracy_score
model = svm.SVC()
f_model = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])