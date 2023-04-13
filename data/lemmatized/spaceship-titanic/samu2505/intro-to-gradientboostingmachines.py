import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import gc
from tqdm.auto import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from catboost import CatBoostClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col=0)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col=0)
_input1.head()
_input1['Transported'].value_counts().plot(kind='bar')
plt.title('Distribution of Transported', fontsize=15)
_input1.dtypes.value_counts().plot(kind='bar')
plt.title('Distribution of data types', fontsize=15)
print('Shape of the training set: ', _input1.shape)
print('Total missing data in training set: ', _input1.isna().sum().sum())
print('\nShape of the test set: ', _input0.shape)
print('Total missing data in test set: ', _input0.isna().sum().sum())
missing_train_data = [(col, _input1[col].isna().sum() / len(_input1) * 100) for col in _input1.columns.tolist() if _input1[col].isna().sum() > 0]
missing_test_data = [(col, _input0[col].isna().sum() / len(_input0) * 100) for col in _input0.columns.tolist() if _input0[col].isna().sum() > 0]
missing_train_data = pd.DataFrame(missing_train_data, columns=['feature', 'MissingPct']).sort_values(by='MissingPct', ascending=False)
missing_test_data = pd.DataFrame(missing_test_data, columns=['feature', 'MissingPct']).sort_values(by='MissingPct', ascending=False)
sns.histplot(x=missing_train_data['MissingPct'], data=missing_train_data, bins=10)
plt.title('Distribution of missing data from the training set', fontsize=15)
sns.histplot(x=missing_test_data['MissingPct'], data=missing_test_data, bins=10)
plt.title('Distribution of missing values from the test set')
numerical_cols = _input0.select_dtypes(include=[np.number, np.bool8]).columns.tolist()
categorical_col = _input0.select_dtypes(include=['object', 'category']).columns.tolist()
target = 'Transported'
imputer_num = SimpleImputer(strategy='mean')
_input1[numerical_cols] = imputer_num.fit_transform(_input1[numerical_cols])
_input0[numerical_cols] = imputer_num.fit_transform(_input0[numerical_cols])
imputer_cat = SimpleImputer(strategy='constant')
_input1[categorical_col] = imputer_cat.fit_transform(_input1[categorical_col])
_input0[categorical_col] = imputer_cat.fit_transform(_input0[categorical_col])

def corr_func(df, features=numerical_cols, target=target):
    corr = df[features].corrwith(df[target])
    return pd.DataFrame({'feature': corr.index, 'correlation': corr.values}).sort_values(by='correlation', ascending=False)
corr_func(_input1)
for col in categorical_col:
    (_input1[col], _) = _input1[col].factorize()
    (_input0[col], _) = _input0[col].factorize()
corr_func(_input1, features=categorical_col)
corr_func(_input1)
features = [*numerical_cols, *categorical_col]
(X_train, X_valid, y_train, y_valid) = train_test_split(_input1[features], _input1[target], test_size=0.1, random_state=1223, shuffle=True)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
ada_clf = AdaBoostClassifier(n_estimators=300, learning_rate=0.265)
hist_clf = HistGradientBoostingClassifier(learning_rate=0.02)
extra_clf = ExtraTreesClassifier(n_estimators=600, max_leaf_nodes=13, max_depth=9)