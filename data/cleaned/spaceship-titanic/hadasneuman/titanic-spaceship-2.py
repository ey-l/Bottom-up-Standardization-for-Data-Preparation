import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
print('Train data:', train_data.shape)
print('Test data:', test_data.shape)
train_data.head()
train_data.describe()
train_data.info()
train_data.groupby(['Transported']).mean()
import matplotlib.pyplot as plt
import seaborn as sns
corrmat = train_data.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
train_data.boxplot(column=['Age'], by='Transported')
train_data.boxplot(column=['Spa'], by='Transported')
train_data.boxplot(column=['RoomService'], by='Transported')
train_data.boxplot(column=['VRDeck'], by='Transported')
X = train_data.copy()
X_test = test_data.copy()
y = X['Transported']
X.drop(['Transported'], axis=1, inplace=True)
all_dfs = [X, X_test]
X_test.head()
X.isnull().sum()
print('X columns:\t', X.columns)
print('X_test columns:\t', X_test.columns)
object_cols = [col for col in X.columns if X[col].dtype == 'object']
good_label_cols = [col for col in object_cols if set(X_test[col]).issubset(set(X[col]))]
print('object_cols', object_cols)
print('good_label_cols', good_label_cols)
for df in all_dfs:
    df.drop(['Cabin', 'Name'], axis=1, inplace=True)
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
numerical_cols_with_missing = [cname for cname in numerical_cols if X[cname].isnull().any()]
print('Numerical columns with missing values:', numerical_cols_with_missing)
categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']
categorical_cols_with_missing = [cname for cname in categorical_cols if X[cname].isnull().any()]
print('Categorial columns with missing values:', categorical_cols_with_missing)
for df in all_dfs:
    for col in numerical_cols_with_missing + categorical_cols_with_missing:
        df[col + '_was_missing'] = df[col].isnull()
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')
num_X = X[numerical_cols]
num_X_test = X_test[numerical_cols]
num_X_imp = pd.DataFrame(num_imputer.fit_transform(num_X))
num_X_test_imp = pd.DataFrame(num_imputer.transform(num_X_test))
cat_X = X[categorical_cols]
cat_X_test = X_test[categorical_cols]
cat_X_imp = pd.DataFrame(cat_imputer.fit_transform(cat_X))
cat_X_test_imp = pd.DataFrame(cat_imputer.transform(cat_X_test))
cat_X_imp.columns = cat_X.columns
cat_X_test_imp.columns = cat_X_test.columns
num_X_imp.columns = num_X.columns
num_X_test_imp.columns = num_X_test.columns
X_imp = pd.concat([cat_X_imp, num_X_imp], axis=1)
X_test_imp = pd.concat([cat_X_test_imp, num_X_test_imp], axis=1)
all_dfs = [X_imp, X_test_imp]
print('X_imp:')
print(X_imp.isnull().sum())
print('X_test_imp')
print(X_test_imp.isnull().sum())
for df in all_dfs:
    df.index = df['PassengerId']
    temp = df['PassengerId'].str.split('_', expand=True)
    df['Group'] = temp.loc[:, 0]
    df['GroupSize'] = df.groupby('Group').Group.transform('count')
    df.drop(['PassengerId', 'Group'], axis=1, inplace=True)
X_imp.head()
categorical_cols = [cname for cname in X_imp.columns if X_imp[cname].dtype == 'object']
low_cardinality_cols = [col for col in categorical_cols if X_imp[col].nunique() < 10]
high_cardinality_cols = [col for col in categorical_cols if X_imp[col].nunique() >= 10]
print('low_cardinality_cols', low_cardinality_cols)
print('high_cardinality_cols', high_cardinality_cols)
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_imp[low_cardinality_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test_imp[low_cardinality_cols]))
OH_cols_train.columns = OH_encoder.get_feature_names_out()
OH_cols_test.columns = OH_encoder.get_feature_names_out()
OH_cols_train.index = X_imp.index
OH_cols_test.index = X_test_imp.index
num_X = X_imp.drop(categorical_cols, axis=1)
num_X_test = X_test_imp.drop(categorical_cols, axis=1)
OH_X = pd.concat([num_X, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
OH_X.head()
OH_X_test.head()
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(OH_X, y, train_size=0.7, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
model_rf = RandomForestClassifier(n_estimators=100, random_state=0)
model_xg = XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=4)