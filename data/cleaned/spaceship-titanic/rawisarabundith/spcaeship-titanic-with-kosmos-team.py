import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('train size:', train.shape)
print('test size:', test.shape)
train.info()
train.describe()
train.head()
train.tail()
null_cols_values = train.isnull().sum()
null_datapoints = null_cols_values[null_cols_values > 0].sum()
print(null_cols_values)
print(f'sum of Missing Value = {null_datapoints}')
train.nunique()

def val_count_BarPlot(data=None, count_cols=None):
    data_count = data[count_cols].value_counts()
    (fig, ax) = plt.subplots(figsize=(8, 6))
    pbar = ax.bar(data_count.index.astype(str), data_count.values)
    ax.set_title(f'{count_cols} values count')

val_count_BarPlot(data=train, count_cols='Transported')
val_count_BarPlot(data=train, count_cols='HomePlanet')
for val in list(train['HomePlanet'].unique()):
    print(f'{val} Transported')
    print(train.loc[train['HomePlanet'] == val]['Transported'].value_counts(), '\n')
val_count_BarPlot(data=train, count_cols='CryoSleep')
for val in list(train['CryoSleep'].unique()):
    print(f'{val} Transported')
    print(train.loc[train['CryoSleep'] == val]['Transported'].value_counts(), '\n')
val_count_BarPlot(data=train, count_cols='Destination')
for val in list(train['Destination'].unique()):
    print(f'{val} Transported')
    print(train.loc[train['Destination'] == val]['Transported'].value_counts(), '\n')
val_count_BarPlot(data=train, count_cols='VIP')
for val in list(train['VIP'].unique()):
    print(f'{val} Transported')
    print(train.loc[train['VIP'] == val]['Transported'].value_counts())
sns.displot(train['Age'])

data_continuous = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for data in data_continuous:
    sns.displot(data=train, x=data)

(fig, ax) = plt.subplots(figsize=(12, 8))
sns.heatmap(train.corr(), annot=True, cmap='plasma')


def split_cabin(data):
    data = data.split('/')
    return data

def split_pid(data):
    data = data.split('_')
    return data
X = train.copy()
categorical_variables = X.select_dtypes(include=['object']).columns
numerical_variable = X._get_numeric_data().columns
for cat_col in categorical_variables:
    X[cat_col] = X[cat_col].fillna(X[cat_col].mode()[0])
for num_col in numerical_variable:
    X[num_col] = X[num_col].fillna(X[num_col].mean())
null = X.isnull().sum()
null
Y = X.pop('Transported')
X[['deck', 'num', 'side']] = list(map(split_cabin, X['Cabin']))
X[['group', 'numInGroup']] = list(map(split_pid, X['PassengerId']))
X['group'] = X['group'].astype(int)
X['numInGroup'] = X['numInGroup'].astype(int)
X['totalBill'] = X[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
X = X.drop(columns=['PassengerId', 'Name'])
X

def checkFamily(data=None):
    for i in range(data.shape[0]):
        if len(data['group'].loc[data['group'] == data['group'].values] >= 3):
            data['family'].values[i] = 1
    return data
X['family'] = np.zeros((8693,), dtype=int)
checkFamily(X)

def checkAge(data=None):
    if data > 20:
        data = 1
    else:
        data = 0
    return data
X['isAdult'] = list(map(checkAge, X['Age'].values))
X
for colname in X.select_dtypes('object'):
    (X[colname], _) = X[colname].factorize()
    X[colname] = X[colname].astype(int)
discrete_features = X.dtypes == int
discrete_features
from sklearn.feature_selection import mutual_info_classif

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, Y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores Clf', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
mi_scores = make_mi_scores(X, Y, discrete_features)
mi_scores[::3]

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
from sklearn.model_selection import train_test_split
categorical_variables = train.select_dtypes(include=['object']).columns
numerical_variable = train._get_numeric_data().columns
for cat_col in categorical_variables:
    train[cat_col] = train[cat_col].fillna(train[cat_col].mode()[0])
for num_col in numerical_variable:
    train[num_col] = train[num_col].fillna(train[num_col].mean())
(X_train, X_valid, Y_train, Y_valid) = train_test_split(train[list(train.columns)[0:13]], train['Transported'], test_size=0.2, random_state=42)

def split_cabin(data):
    data = data.split('/')
    return data

def split_pid(data):
    data = data.split('_')
    return data

def checkAge(data=None):
    if data > 20:
        data = 1
    else:
        data = 0
    return data
X_train = X_train.reset_index(drop=True)
X_train['totalBill'] = X_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
X_train[['deck', 'num', 'side']] = list(map(split_cabin, X_train['Cabin']))
X_train['num'] = X_train['num'].astype(int)
X_train[['group', 'numInGroup']] = list(map(split_pid, X_train['PassengerId']))
X_train['group'] = X_train['group'].astype(int)
X_train['numInGroup'] = X_train['numInGroup'].astype(int)
X_train['isAdult'] = list(map(checkAge, X_train['Age'].values))
X_train = X_train.drop(columns=['PassengerId', 'Name', 'Cabin'])
s = X_train.dtypes == 'object'
cat_cols = list(s[s].index)
cat_cols
prepared_X_train = X_train.copy()
for colname in prepared_X_train.select_dtypes('object'):
    (prepared_X_train[colname], _) = prepared_X_train[colname].factorize()
    prepared_X_train[colname] = prepared_X_train[colname].astype(int)
discrete_features = prepared_X_train.dtypes == int
X_valid = X_valid.reset_index(drop=True)
X_valid['totalBill'] = X_valid[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
X_valid[['deck', 'num', 'side']] = list(map(split_cabin, X_valid['Cabin']))
X_valid['num'] = X_valid['num'].astype(int)
X_valid[['group', 'numInGroup']] = list(map(split_pid, X_valid['PassengerId']))
X_valid['group'] = X_valid['group'].astype(int)
X_valid['numInGroup'] = X_valid['numInGroup'].astype(int)
X_valid['isAdult'] = list(map(checkAge, X_valid['Age'].values))
X_valid = X_valid.drop(columns=['PassengerId', 'Name', 'Cabin'])
s = X_valid.dtypes == 'object'
cat_cols = list(s[s].index)
cat_cols
prepared_X_valid = X_valid.copy()
for colname in prepared_X_valid.select_dtypes('object'):
    (prepared_X_valid[colname], _) = prepared_X_valid[colname].factorize()
    prepared_X_valid[colname] = prepared_X_valid[colname].astype(int)
discrete_features = prepared_X_valid.dtypes == int
train = train.reset_index(drop=True)
train['totalBill'] = train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
train[['deck', 'num', 'side']] = list(map(split_cabin, train['Cabin']))
train['num'] = train['num'].astype(int)
train[['group', 'numInGroup']] = list(map(split_pid, train['PassengerId']))
train['group'] = train['group'].astype(int)
train['numInGroup'] = train['numInGroup'].astype(int)
train['isAdult'] = list(map(checkAge, train['Age'].values))
train = train.drop(columns=['PassengerId', 'Name', 'Cabin'])
s = train.dtypes == 'object'
cat_cols = list(s[s].index)
cat_cols
prepared_train = train.copy()
for colname in prepared_train.select_dtypes('object'):
    (prepared_train[colname], _) = prepared_train[colname].factorize()
    prepared_train[colname] = prepared_train[colname].astype(int)
discrete_features = prepared_train.dtypes == int
categorical_variables = test.select_dtypes(include=['object']).columns
numerical_variable = test._get_numeric_data().columns
for cat_col in categorical_variables:
    test[cat_col] = test[cat_col].fillna(test[cat_col].mode()[0])
for num_col in numerical_variable:
    test[num_col] = test[num_col].fillna(test[num_col].mean())
test['totalBill'] = test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
test['CryoSleep'] = test['CryoSleep'].astype('object')
test['VIP'] = test['VIP'].astype('object')
test[['deck', 'num', 'side']] = list(map(split_cabin, test['Cabin']))
test['num'] = test['num'].astype(int)
test[['group', 'numInGroup']] = list(map(split_pid, test['PassengerId']))
test['group'] = test['group'].astype(int)
test['numInGroup'] = test['numInGroup'].astype(int)
test['isAdult'] = list(map(checkAge, test['Age'].values))
test = test.drop(columns=['PassengerId', 'Name', 'Cabin'])
s = test.dtypes == 'object'
cat_cols = list(s[s].index)
cat_cols
prepared_test = test.copy()
for colname in prepared_test.select_dtypes('object'):
    (prepared_test[colname], _) = prepared_test[colname].factorize()
    prepared_test[colname] = prepared_test[colname].astype(int)
discrete_features = prepared_test.dtypes == int
from sklearn import metrics

def modelEvaluation(Y_test=None, predictions=None):
    accuracy = metrics.accuracy_score(Y_test, predictions)
    precision = metrics.precision_score(Y_test, predictions)
    recall = metrics.recall_score(Y_test, predictions)
    f1 = metrics.f1_score(Y_test, predictions)
    ROC_AUC_Score = metrics.roc_auc_score(Y_test, predictions)
    print(f'Accuracy = {np.round(accuracy, 4)}')
    print(f'Precision = {np.round(precision, 4)}')
    print(f'recall = {np.round(recall, 4)}')
    print(f'F1 = {np.round(f1, 4)}')
    print(f'ROC AUC Score = {np.round(ROC_AUC_Score, 4)}')
from sklearn.ensemble import RandomForestClassifier
rf_CLF = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
train.isnull().sum()