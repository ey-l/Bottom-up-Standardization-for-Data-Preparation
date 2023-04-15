import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
to_predict = pd.read_csv('data/input/spaceship-titanic/test.csv')
to_predict.isna().sum()
train.head()



train = train.astype({'VIP': 'bool', 'CryoSleep': 'bool'})
to_predict = to_predict.astype({'VIP': 'bool', 'CryoSleep': 'bool'})
import missingno as msno
import matplotlib.pyplot as plt
msno.heatmap(train)
s = train.dtypes == 'object'
col_objects = list(s[s].index)
for col in col_objects:
    print(train[col].value_counts())
train[['Cabin_A', 'Cabin_B', 'Cabin_C']] = train['Cabin'].str.split('/', expand=True)
to_predict[['Cabin_A', 'Cabin_B', 'Cabin_C']] = to_predict['Cabin'].str.split('/', expand=True)
train = train.drop('Cabin', axis=1)
to_predict = to_predict.drop('Cabin', axis=1)
print(train.dtypes, to_predict.dtypes)
s = train.dtypes == 'object'
col_objects = list(s[s].index)
train[col_objects] = train[col_objects].fillna('Unknown')
to_predict[col_objects] = to_predict[col_objects].fillna('Unknown')
train.sample(20)
pd.concat([pd.DataFrame(train.isna().sum()), pd.DataFrame(train.dtypes)], axis=1)
n = train.dtypes == 'float'
col_nums = list(n[n].index)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_num_columns(df):
    (fig, ax) = plt.subplots(1, len(col_nums), figsize=(15, 8), tight_layout=True)
    for (i, col) in enumerate(col_nums):
        ax[i] = sns.violinplot(data=df, y=col, ax=ax[i])
plot_num_columns(train)
num_non_age = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

def find_upper_outlier(col):
    Q3 = np.quantile(col, 0.75)
    Q1 = np.quantile(col, 0.25)
    limit_finder = 1.5 * (Q3 - Q1)
    upper_limit = Q3 + limit_finder
    return upper_limit
for col in col_nums:
    median = np.quantile(train[col], 0.5)
    train[col] = train[col].fillna(median)
    to_predict[col] = to_predict[col].fillna(median)
for col in col_nums:
    median = np.median(train[~train[col].isna()][col])
    train[col] = train[col].fillna(median)
    to_predict[col] = to_predict[col].fillna(median)
print(train.isna().sum(), to_predict.isna().sum())
sns.countplot(data=train, x='Transported')

def obj_is_transported_by_group(df, col):
    grouped = df.groupby(col).Transported.mean().sort_values(ascending=False).head(30)
    (fig, ax) = plt.subplots(figsize=(8, 7))
    ax = grouped.plot(kind='bar', label=col)
    plt.title('Percentage of Transported by ' + col)
    plt.xticks(rotation=45)

    return ax

def num_is_transported_by_group(df, col, choice):
    (fig, ax) = plt.subplots(figsize=(8, 7))
    if choice == 'violin':
        ax = sns.violinplot(data=df, y=col, x='Transported')
    if choice == 'box':
        ax = sns.boxplot(data=df, y=col, x='Transported')
    plt.xticks(rotation=45)

    return ax
for col in col_objects:
    obj_is_transported_by_group(train, col)
for col in col_nums:
    num_is_transported_by_group(train, col, 'violin')
for col in col_nums:
    num_is_transported_by_group(train, col, 'box')
light_pallete = sns.light_palette('steelblue', as_cmap=True)
plt.figure(figsize=(10, 10))
sns.heatmap(train.corr(), cmap=light_pallete, annot=True)
train.head()
X = train.drop(['Transported', 'Name', 'PassengerId'], axis=1)
y = train['Transported']
to_predict_X = to_predict.drop(['Name', 'PassengerId'], axis=1)
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
print(X_train.shape, y_train.shape)
ohe = OneHotEncoder()
rs = RobustScaler(quantile_range=(0, 10))
scaler = StandardScaler()
preprocessor = ColumnTransformer(transformers=[('cat', ohe, ['HomePlanet', 'Destination', 'Cabin_A', 'Cabin_C']), ('remove_outlier', rs, num_non_age), ('scaler', scaler, col_nums)])
dt = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', dt)])
params_dt = {'classifier__criterion': ['gini', 'entropy'], 'classifier__min_samples_leaf': [0.04, 0.06, 0.08], 'classifier__max_features': [0.2, 0.4, 0.6, 0.8, 'auto'], 'classifier__max_depth': [3, 4, 5, 6, 7]}
grid_dt = GridSearchCV(estimator=pipeline, param_grid=params_dt, scoring='accuracy', cv=10, n_jobs=-1)