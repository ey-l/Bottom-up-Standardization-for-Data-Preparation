import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
if not sys.warnoptions:
    warnings.simplefilter('ignore')
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
plt.figure(figsize=(17, 10))
corr = _input1.corr()
ax = sns.heatmap(corr)
train_features = [col for col in _input1.columns if _input1[col].isna().sum() == 0]
test_features = [col for col in _input0.columns if _input0[col].isna().sum() == 0]
train_df = _input1[train_features].copy()
test_df = _input0[test_features].copy()
train_df = train_df.drop(columns=['Id'], axis=1)
test_df = test_df.drop(columns=['Id'], axis=1)
(train_df.shape[1], test_df.shape[1])
features_1 = [col for col in train_df.columns]
features_2 = [col for col in test_df.columns]
features = [col for col in features_1 if col in features_2]
train_df = train_df[features]
(train_df.shape[1], test_df.shape[1])
print([col for col in test_df.columns if col not in train_df.columns][0])
test_df = test_df.drop(columns=['Electrical'], axis=1)

def encoder(df):
    """This is a specific function."""
    le = LabelEncoder()
    obj_col = [col for col in df.columns if df[col].dtypes == 'object']
    for col in obj_col:
        df[col] = le.fit_transform(df[col])
    print('Success Encode')
encoder(train_df)
encoder(test_df)
X = train_df
y = _input1.loc[:, ['SalePrice']]
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=0)
rfc = RandomForestRegressor()
scaler = StandardScaler()
pipe = Pipeline(steps=[('scaler', scaler), ('rfc', rfc)])
parameters = {'rfc__n_estimators': [13, 50, 134, 200], 'rfc__max_depth': [5, 10, 23, 100], 'rfc__random_state': [0]}
g_search = GridSearchCV(pipe, param_grid=parameters, cv=3, n_jobs=1)