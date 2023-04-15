import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.plotting.register_matplotlib_converters()

train_raw = 'https://raw.githubusercontent.com/devthumos/spaceship_titanic/master/train.csv'
test_raw = 'https://raw.githubusercontent.com/devthumos/spaceship_titanic/master/test.csv'
train_set = pd.read_csv(train_raw)
test_set = pd.read_csv(test_raw)
train_set.head()
train_set.columns
train_set.info()
import re
train_set['Deck'] = train_set.Cabin.apply(lambda x: re.split('/', str(x))[0] if len(re.split('/', str(x))) > 2 else x)
train_set['Side'] = train_set.Cabin.apply(lambda x: re.split('/', str(x))[2] if len(re.split('/', str(x))) > 2 else x)
train_set = train_set.drop('Cabin', axis=1)
test_set['Deck'] = test_set.Cabin.apply(lambda x: re.split('/', str(x))[0] if len(re.split('/', str(x))) > 2 else x)
test_set['Side'] = test_set.Cabin.apply(lambda x: re.split('/', str(x))[2] if len(re.split('/', str(x))) > 2 else x)
test_set = test_set.drop('Cabin', axis=1)
train_set['Spent'] = train_set.RoomService + train_set.FoodCourt + train_set.ShoppingMall + train_set.Spa + train_set.VRDeck
test_set['Spent'] = test_set.RoomService + test_set.FoodCourt + test_set.ShoppingMall + test_set.Spa + test_set.VRDeck
categorical_columns = [column for column in train_set.columns if train_set[column].dtype == 'object']
numerical_columns = [column for column in train_set.columns if train_set[column].dtype not in ['object', 'bool']]
bool_columns = list(set(train_set.columns) - (set(categorical_columns) | set(numerical_columns)))
print(categorical_columns)
print(numerical_columns)
print(bool_columns)
train_set[categorical_columns].isnull().sum()
test_set[categorical_columns].isnull().sum()
train_set[numerical_columns].isnull().sum()
test_set[numerical_columns].isnull().sum()
train_set[bool_columns].isnull().sum()
high_columns = [column for column in categorical_columns if train_set[column].nunique() > 15]
for column in high_columns:
    categorical_columns.remove(column)
high_columns
train_set_index = train_set.PassengerId
test_set_index = test_set.PassengerId
train_set = train_set.drop(high_columns, axis=1)
test_set = test_set.drop(high_columns, axis=1)
from sklearn.model_selection import train_test_split
X = train_set.drop('Transported', axis=1)
Y = train_set.Transported
(X_train, X_val, Y_train, Y_val) = train_test_split(X, Y, test_size=0.15, shuffle=True, stratify=Y, random_state=0)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')), ('onehotencoding', OneHotEncoder(handle_unknown='ignore', sparse=False))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_columns), ('cat', categorical_transformer, categorical_columns)])
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])