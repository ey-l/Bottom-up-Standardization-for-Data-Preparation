import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
train_data.sample(20)
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
test_data.sample(20)
import matplotlib.pyplot as plt

import seaborn as sns
train_data.columns
train_data.info()
train_data.isna().sum() / train_data.shape[0]
sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False)
train_data.describe()
train_data.HomePlanet.value_counts()
sns.countplot(x='HomePlanet', data=train_data)
train_data.Destination.value_counts()
sns.countplot(x='Destination', data=train_data)
train_data.CryoSleep.value_counts()
sns.countplot(x='CryoSleep', data=train_data)
train_data.Transported.value_counts()
sns.countplot(x='Transported', data=train_data)
sns.heatmap(train_data.corr(method='pearson'), annot=True)
train_data.groupby(['Transported', 'HomePlanet']).agg({'Transported': 'count'})
sns.catplot(x='Transported', col='HomePlanet', kind='count', data=train_data)
train_data.groupby(['Transported', 'Destination']).agg({'Transported': 'count'})
sns.catplot(x='Transported', col='Destination', kind='count', data=train_data)
train_data.groupby(['Transported', 'CryoSleep']).agg({'Transported': 'count'})
sns.catplot(x='Transported', col='CryoSleep', kind='count', data=train_data)
y_train = train_data.Transported
X_train_full = train_data.drop(['Transported'], axis=1, inplace=False)
X_test_full = test_data
X_train_full
X_train_full.isna().sum()
X_test_full.isna().sum()
list(X_train_full['Name'][:200])
X_train_full[['FirstName', 'LastName']] = X_train_full['Name'].str.split(' ', expand=True, n=1)
X_train_full.drop(columns=['Name'], inplace=True)
X_train_full
X_train_full['LastName'].nunique()
X_train_full['LastName'].isna().sum()
X_train_full['FirstName'].fillna(value='Unknown', inplace=True)
X_train_full['LastName'].fillna(value='Unknown', inplace=True)
X_test_full[['FirstName', 'LastName']] = X_test_full['Name'].str.split(' ', expand=True, n=1)
X_test_full.drop(columns=['Name'], inplace=True)
X_test_full
X_test_full['FirstName'].fillna(value='Unknown', inplace=True)
X_test_full['LastName'].fillna(value='Unknown', inplace=True)
X_test_full['LastName'].nunique()
len(set(X_train_full['LastName']).intersection(set(X_test_full['LastName'])))
X_all = pd.concat([X_train_full, X_test_full])
X_all
family_size = X_all['LastName'].value_counts()
family_size
import statistics
family_size['Unknown'] = statistics.median(family_size.values)
family_size['Unknown']
X_train_full['FamilySize'] = X_train_full['LastName'].apply(lambda s: family_size[s])
X_train_full
X_test_full['FamilySize'] = X_test_full['LastName'].apply(lambda s: family_size[s])
X_test_full
X_train_full[['Deck', 'Num', 'Side']] = X_train_full['Cabin'].str.split('/', expand=True)
X_test_full[['Deck', 'Num', 'Side']] = X_test_full['Cabin'].str.split('/', expand=True)
X_train_full
X_train = X_train_full.drop(columns=['FirstName', 'LastName', 'Cabin', 'Num'])
X_test = test_data.drop(columns=['FirstName', 'LastName', 'Cabin', 'Num'])
cat_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
cat_cols
num_cols = [col for col in X_train.columns if X_train[col].dtype == 'float64']
num_cols
luxury_cols = [col for col in num_cols if col != 'Age']
luxury_cols
for col in cat_cols:
    X_train[col].fillna(value=X_train[col].mode()[0], inplace=True)
    X_test[col].fillna(value=X_train[col].mode()[0], inplace=True)
X_train.isna().sum()
for col in luxury_cols:
    print(X_train.loc[X_train.CryoSleep == True, col].max())
for col in luxury_cols:
    print(X_train.loc[X_train.CryoSleep == False, col].min(), X_train.loc[X_train.CryoSleep == False, col].median())
median_costs = X_train.loc[X_train.CryoSleep == False, luxury_cols].median()
median_costs
for col in luxury_cols:
    X_train[col].fillna(X_train.groupby('CryoSleep')[col].transform('median'), inplace=True)
    X_test[col].fillna(X_test.groupby('CryoSleep')[col].transform('median'), inplace=True)
X_train.isna().sum()
X_train.Age.fillna(value=X_train.Age.median(), inplace=True)
X_test.Age.fillna(value=X_train.Age.median(), inplace=True)
X_train.isna().sum()
X_test.isna().sum()
from sklearn.feature_selection import mutual_info_classif

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(['object', 'category']):
        (X[colname], _) = X[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')
make_mi_scores(X_train, y_train)
X_train = pd.get_dummies(X_train)
X_train
X_test = pd.get_dummies(X_test)
X_test
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
best_parameters = (0, 0)
best_score = 0
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for k in range(100, 500, 100):
    for d in range(5, 20, 5):
        print(f'Number of trees = {k}, max depth = {d}')
        clf = RandomForestClassifier(n_estimators=k, max_depth=d, random_state=42)
        score = round(cross_val_score(clf, X_train, y_train, cv=kf, scoring='accuracy').mean(), 3)
        print(f'Accuracy = {score}')
        if score > best_score:
            best_score = score
            best_parameters = (k, d)
best_parameters
model = RandomForestClassifier(n_estimators=best_parameters[0], max_depth=best_parameters[1], random_state=42)