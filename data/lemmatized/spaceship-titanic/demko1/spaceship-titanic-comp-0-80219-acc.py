import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from yellowbrick.cluster import KElbowVisualizer
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVC
import optuna
df = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId').reset_index(drop=True)
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId').reset_index(drop=True)
df.head()
print('Shape ')
print()
print('Train: ', df.shape)
print('Test: ', df_test.shape)
print('NaN values ')
print()
print('Train: \n', df.isna().sum(), '\n')
print('Test: \n', df_test.isna().sum())
df['Transported'] = df['Transported'].astype(np.int8)
df.head()
df.Transported.value_counts()
cabin_splited = df['Cabin'].str.split('/', expand=True)
cabin_splited.columns = ['Cabin_deck', 'Cabin_num', 'Cabin_side']
cabin_splited.head()
df1 = pd.concat([df, cabin_splited], axis=1).drop('Cabin', axis=1)
df1.head()
cabin_splited = df_test['Cabin'].str.split('/', expand=True)
cabin_splited.columns = ['Cabin_deck', 'Cabin_num', 'Cabin_side']
df_test_1 = pd.concat([df_test, cabin_splited], axis=1).drop('Cabin', axis=1)
df_test_1.head()
object_cols = [i for i in df1.columns if df1[i].dtype == 'O']
for i in object_cols:
    print(i, ': ', df1[i].nunique())
df2 = df1.drop(['Name', 'Cabin_num'], axis=1)
df_test_2 = df_test_1.drop(['Name', 'Cabin_num'], axis=1)
df2.head()
object_cols = [i for i in df2.columns if df2[i].dtype == 'O']
df2[object_cols] = df2[object_cols].fillna(-1)
df_test_2[object_cols] = df_test_2[object_cols].fillna(-1)
df2.isna().sum()
cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df[df[cost_cols].isna().any(axis=1)]
df3 = df2.copy()
df_test_3 = df_test_2.copy()
df3[cost_cols] = df3[cost_cols].fillna(0.0)
df_test_3[cost_cols] = df_test_3[cost_cols].fillna(0.0)
df3.head(8)
df3.isna().sum()
df3['Age'].hist(bins=20, rwidth=0.8)
df3.Age.value_counts()
df4 = df3.copy()
df_test_4 = df_test_3.copy()
df4['Age'] = df4['Age'].fillna(24.0)
df_test_4['Age'] = df_test_4['Age'].fillna(24.0)
df4.isna().sum()
for i in cost_cols:
    plt.subplots()
    sns.distplot(df[i])
df4[cost_cols].describe()
df4.head()
obj_cols = [i for i in df4.columns if df4[i].dtype == 'O']
for i in obj_cols:
    print(df4[i].unique())
for i in obj_cols:
    print(df_test_4[i].unique())
df5 = pd.concat([df4, pd.get_dummies(df4[obj_cols], drop_first=True)], axis=1).drop(obj_cols, axis=1)
df_test_5 = pd.concat([df_test_4, pd.get_dummies(df_test_4[obj_cols], drop_first=True)], axis=1).drop(obj_cols, axis=1)
df5.head()
df6 = df5.copy()
df_test_6 = df_test_5.copy()
df6['SumOfPay'] = df6[cost_cols].sum(axis=1)
df_test_6['SumOfPay'] = df_test_6[cost_cols].sum(axis=1)
df6
sns.heatmap(df6[cost_cols].corr(), annot=True)
df7 = df6.copy()
df_test_7 = df_test_6.copy()
df7['SumOfPay_per_Age'] = (df7.SumOfPay / df7.Age).replace(np.inf, 0).fillna(0.0)
df7.head()
df7 = df7[(df7.SumOfPay_per_Age - df7.SumOfPay_per_Age.mean()) / df7.SumOfPay_per_Age.std() < 3]
df7.head()
df7 = df7.drop('SumOfPay_per_Age', axis=1)
df7
X = df6.copy()
y = X.pop('Transported')
mi = mutual_info_classif(X, y)
mi
mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)
mi.head(10)
model = XGBClassifier(n_estimators=100, random_state=0)
scores = cross_val_score(model, X, y, cv=10)
scores
np.mean(scores)
0.8029502797508036 < 0.801108289353597
scores
model = RandomForestClassifier(n_estimators=100, random_state=0)
scores = cross_val_score(model, X, y, cv=10)
scores
np.mean(scores)
scores

def run(trial):
    param = {'n_estimators': trial.suggest_int('n_estimators', 100, 500), 'max_depth': trial.suggest_int('max_depth', 2, 25), 'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0), 'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0), 'min_child_weight': trial.suggest_float('min_child_weight', 0.0, 5.0), 'gamma': trial.suggest_int('gamma', 0, 5), 'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5), 'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.01), 'nthread': -1, 'random_state': 0}
    (X_train, X_val, y_train, y_val) = train_test_split(X, y)
    model1 = XGBClassifier(**param)