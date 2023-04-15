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

train_df = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col=0)
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col=0)
train_df.head()
train_df['Transported'].value_counts().plot(kind='bar')
plt.title('Distribution of Transported', fontsize=15)
train_df.dtypes.value_counts().plot(kind='bar')
plt.title('Distribution of data types', fontsize=15)
print('Shape of the training set: ', train_df.shape)
print('Total missing data in training set: ', train_df.isna().sum().sum())
print('\nShape of the test set: ', test_df.shape)
print('Total missing data in test set: ', test_df.isna().sum().sum())
missing_train_data = [(col, train_df[col].isna().sum() / len(train_df) * 100) for col in train_df.columns.tolist() if train_df[col].isna().sum() > 0]
missing_test_data = [(col, test_df[col].isna().sum() / len(test_df) * 100) for col in test_df.columns.tolist() if test_df[col].isna().sum() > 0]
missing_train_data = pd.DataFrame(missing_train_data, columns=['feature', 'MissingPct']).sort_values(by='MissingPct', ascending=False)
missing_test_data = pd.DataFrame(missing_test_data, columns=['feature', 'MissingPct']).sort_values(by='MissingPct', ascending=False)
sns.histplot(x=missing_train_data['MissingPct'], data=missing_train_data, bins=10)
plt.title('Distribution of missing data from the training set', fontsize=15)
sns.histplot(x=missing_test_data['MissingPct'], data=missing_test_data, bins=10)
plt.title('Distribution of missing values from the test set')
numerical_cols = test_df.select_dtypes(include=[np.number, np.bool8]).columns.tolist()
categorical_col = test_df.select_dtypes(include=['object', 'category']).columns.tolist()
target = 'Transported'
imputer_num = SimpleImputer(strategy='mean')
train_df[numerical_cols] = imputer_num.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = imputer_num.fit_transform(test_df[numerical_cols])
imputer_cat = SimpleImputer(strategy='constant')
train_df[categorical_col] = imputer_cat.fit_transform(train_df[categorical_col])
test_df[categorical_col] = imputer_cat.fit_transform(test_df[categorical_col])

def corr_func(df, features=numerical_cols, target=target):
    corr = df[features].corrwith(df[target])
    return pd.DataFrame({'feature': corr.index, 'correlation': corr.values}).sort_values(by='correlation', ascending=False)
corr_func(train_df)
for col in categorical_col:
    (train_df[col], _) = train_df[col].factorize()
    (test_df[col], _) = test_df[col].factorize()
corr_func(train_df, features=categorical_col)
corr_func(train_df)
features = [*numerical_cols, *categorical_col]
(X_train, X_valid, y_train, y_valid) = train_test_split(train_df[features], train_df[target], test_size=0.1, random_state=1223, shuffle=True)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
ada_clf = AdaBoostClassifier(n_estimators=300, learning_rate=0.265)
hist_clf = HistGradientBoostingClassifier(learning_rate=0.02)
extra_clf = ExtraTreesClassifier(n_estimators=600, max_leaf_nodes=13, max_depth=9)