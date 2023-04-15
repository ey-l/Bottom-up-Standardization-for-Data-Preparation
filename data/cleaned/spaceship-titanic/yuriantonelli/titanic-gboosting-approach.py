import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
training = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
sub = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
training.head()
training.shape
print(training.Name.nunique(), training.Cabin.nunique())
training = training.drop(['Name', 'Cabin'], axis=1)
test = test.drop(['Name', 'Cabin'], axis=1)
training.head()
test.head()
training.isna().sum()
test.isna().sum()
numerical_cols = []
for cname in training.columns:
    if training[cname].dtype in ['int64', 'float64']:
        numerical_cols.append(cname)
training[numerical_cols].describe()
training[numerical_cols].isna().sum()
sns.histplot(data=training, x='Age')
plt.xlim(0, 90)
print(training['Age'].median(), training['Age'].mode())
sns.histplot(data=training, x='RoomService')
plt.xlim(0, 100)
print(training['RoomService'].median(), training['RoomService'].mode())
sns.histplot(data=training, x='FoodCourt')
plt.xlim(0, 100)
print(training['FoodCourt'].median(), training['FoodCourt'].mode())
sns.histplot(data=training, x='ShoppingMall')
plt.xlim(0, 100)
print(training['ShoppingMall'].median(), training['ShoppingMall'].mode())
sns.histplot(data=training, x='Spa')
plt.xlim(0, 100)
print(training['Spa'].median(), training['Spa'].mode())
sns.histplot(data=training, x='VRDeck')
plt.xlim(0, 100)
print(training['VRDeck'].median(), training['VRDeck'].mode())
for i in numerical_cols:
    if i == 'Age':
        training[i].fillna(training[i].mean(skipna=True), inplace=True)
        test[i].fillna(training[i].mean(skipna=True), inplace=True)
    else:
        training[i].fillna(training[i].median(skipna=True), inplace=True)
        test[i].fillna(training[i].median(skipna=True), inplace=True)
training.isna().sum()
test.isna().sum()
training = training.dropna()
training.shape
PassengerId = test.PassengerId
y = training.Transported
X = training.drop(['Transported', 'PassengerId'], axis=1)
test = test.drop(['PassengerId'], axis=1)
X.isna().sum()
test.isna().sum()
categorical_cols = []
for i in X.columns:
    if i not in numerical_cols:
        categorical_cols.append(i)
print(categorical_cols)
print(X[categorical_cols].head(), X[categorical_cols].nunique())
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X[categorical_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(test[categorical_cols]))
OH_cols_train.index = X.index
OH_cols_test.index = test.index
num_X = X.drop(categorical_cols, axis=1)
num_test = test.drop(categorical_cols, axis=1)
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