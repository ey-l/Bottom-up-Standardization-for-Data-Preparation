import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.head()
_input1.shape
print(_input1.Name.nunique(), _input1.Cabin.nunique())
_input1 = _input1.drop(['Name', 'Cabin'], axis=1)
_input0 = _input0.drop(['Name', 'Cabin'], axis=1)
_input1.head()
_input0.head()
_input1.isna().sum()
_input0.isna().sum()
numerical_cols = []
for cname in _input1.columns:
    if _input1[cname].dtype in ['int64', 'float64']:
        numerical_cols.append(cname)
_input1[numerical_cols].describe()
_input1[numerical_cols].isna().sum()
sns.histplot(data=_input1, x='Age')
plt.xlim(0, 90)
print(_input1['Age'].median(), _input1['Age'].mode())
sns.histplot(data=_input1, x='RoomService')
plt.xlim(0, 100)
print(_input1['RoomService'].median(), _input1['RoomService'].mode())
sns.histplot(data=_input1, x='FoodCourt')
plt.xlim(0, 100)
print(_input1['FoodCourt'].median(), _input1['FoodCourt'].mode())
sns.histplot(data=_input1, x='ShoppingMall')
plt.xlim(0, 100)
print(_input1['ShoppingMall'].median(), _input1['ShoppingMall'].mode())
sns.histplot(data=_input1, x='Spa')
plt.xlim(0, 100)
print(_input1['Spa'].median(), _input1['Spa'].mode())
sns.histplot(data=_input1, x='VRDeck')
plt.xlim(0, 100)
print(_input1['VRDeck'].median(), _input1['VRDeck'].mode())
for i in numerical_cols:
    if i == 'Age':
        _input1[i] = _input1[i].fillna(_input1[i].mean(skipna=True), inplace=False)
        _input0[i] = _input0[i].fillna(_input1[i].mean(skipna=True), inplace=False)
    else:
        _input1[i] = _input1[i].fillna(_input1[i].median(skipna=True), inplace=False)
        _input0[i] = _input0[i].fillna(_input1[i].median(skipna=True), inplace=False)
_input1.isna().sum()
_input0.isna().sum()
_input1 = _input1.dropna()
_input1.shape
PassengerId = _input0.PassengerId
y = _input1.Transported
X = _input1.drop(['Transported', 'PassengerId'], axis=1)
_input0 = _input0.drop(['PassengerId'], axis=1)
X.isna().sum()
_input0.isna().sum()
categorical_cols = []
for i in X.columns:
    if i not in numerical_cols:
        categorical_cols.append(i)
print(categorical_cols)
print(X[categorical_cols].head(), X[categorical_cols].nunique())
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X[categorical_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(_input0[categorical_cols]))
OH_cols_train.index = X.index
OH_cols_test.index = _input0.index
num_X = X.drop(categorical_cols, axis=1)
num_test = _input0.drop(categorical_cols, axis=1)
OH_X = pd.concat([num_X, OH_cols_train], axis=1)
OH_test = pd.concat([num_test, OH_cols_test], axis=1)
OH_X.head()
OH_test.head()
print(OH_X.shape, y.shape)
print(OH_test.shape)
mean_list = []
std_list = []
for n in [10, 50, 100, 200]:
    cross_score = cross_val_score(XGBClassifier(n_estimators=n, random_state=0), X=OH_X, y=y, scoring='accuracy')
    print(f'XGBoost WITH {n} TREES')
    print(f'cross_score: {cross_score}')
    print(f'mean = {cross_score.mean()}, standard deviation = {cross_score.std()}')
    mean_list.append(cross_score.mean())
    std_list.append(cross_score.mean())
    print('-----------------------------------------')
df_rf = pd.DataFrame(data={'mean': mean_list}, index=['d_10', 'd_50', 'd_100', 'd_200'])
print(df_rf.sort_values(by=['mean'], ascending=False))
model = XGBClassifier(n_estimators=10)