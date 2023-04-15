import numpy as np
import pandas as pd
import seaborn as sns
import pandas_profiling
from xgboost import XGBClassifier
from sklearn.base import TransformerMixin
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from matplotlib import pyplot as plt
sns.set()
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
X_train = train_df.iloc[:, :-1]
Y_train = train_df.iloc[:, -1]
X_test = test_df
train_df.info()
train_df.head(10)
pandas_profiling.ProfileReport(train_df)
pandas_profiling.ProfileReport(test_df)
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
(fig, axes) = plt.subplots(4, 1, figsize=(15, 30))
for (i, col) in enumerate(cat_cols):
    axis = axes[i]
    col_data_train = train_df[col].value_counts().to_frame() / len(train_df)
    col_data_train['Set'] = 'Train'
    col_data_test = test_df[col].value_counts().to_frame() / len(test_df)
    col_data_test['Set'] = 'Test'
    col_data = pd.concat([col_data_train, col_data_test])
    col_data.index = map(str, col_data.index)
    sns.barplot(data=col_data, x=col_data.index, y=col, hue='Set', ax=axis)
    axis.set(ylabel=None, title=col)
(fig, axes) = plt.subplots(4, 1, figsize=(15, 30))
for (i, col) in enumerate(cat_cols):
    axis = axes[i]
    col_data_trans = train_df[col][Y_train == 0].value_counts().to_frame()
    col_data_trans['Target'] = 'Transpoted'
    col_data_not = train_df[col][Y_train == 1].value_counts().to_frame()
    col_data_not['Target'] = 'Not transported'
    col_data = pd.concat([col_data_trans, col_data_not])
    col_data.index = map(str, col_data.index)
    sns.barplot(data=col_data, x=col_data.index, y=col, hue='Target', ax=axis)
    axis.set(ylabel=None, title=col)
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axes) = plt.subplots(6, 1, figsize=(15, 30))
for (i, col) in enumerate(num_cols[:]):
    axis = axes[i]
    log_scale = (i > 0) * 10
    col_data = pd.concat([train_df[col], test_df[col]], axis=1, ignore_index=True)
    if log_scale:
        col_data.replace({0: np.nan}, inplace=True)
        sns.histplot(data=col_data, ax=axis, kde=True, log_scale=log_scale)
    else:
        sns.histplot(data=col_data, ax=axis, kde=True, bins=40)
    axis.legend(['Train', 'Test'])
money_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df = np.log1p(X_train[money_cols]) / np.log(1000)
df['Transported'] = Y_train.copy()
sns.set(rc={'figure.figsize': (15, 10)})
sns.heatmap(df.corr(), cmap='coolwarm', vmin=-1, vmax=1)
X_train.drop(['Name'], axis=1, inplace=True)
X_test.drop(['Name'], axis=1, inplace=True)
X_train['SummaryExpenses'] = X_train[money_cols].sum(axis=1)
X_test['SummaryExpenses'] = X_test[money_cols].sum(axis=1)
X_train[money_cols] = X_train.loc[:, money_cols].div(X_train['SummaryExpenses'], axis=0)
X_test[money_cols] = X_test.loc[:, money_cols].div(X_test['SummaryExpenses'], axis=0)
X_train['SummaryExpenses'] = np.log1p(X_train['SummaryExpenses']) / np.log(1000)
X_test['SummaryExpenses'] = np.log1p(X_test['SummaryExpenses']) / np.log(1000)
X_train[money_cols] = X_train.loc[:, money_cols].fillna(0)
X_test[money_cols] = X_test.loc[:, money_cols].fillna(0)
X_train['SummaryExpenses'].fillna(0, inplace=True)
X_test['SummaryExpenses'].fillna(0, inplace=True)
X_train['VIP'].fillna(False, inplace=True)
X_test['VIP'].fillna(False, inplace=True)
X_train['CryoSleep'].fillna(X_train['SummaryExpenses'] == 0, inplace=True)
X_test['CryoSleep'].fillna(X_test['SummaryExpenses'] == 0, inplace=True)
mean = X_train['Age'].mean()
X_train['Age'].fillna(mean, inplace=True)
X_test['Age'].fillna(mean, inplace=True)

def id_parser(row):
    s = row['PassengerId']
    (group, _) = s.split('_')
    return int(group)
new_col = 'GroupNumber'
X_train[new_col] = X_train.apply(id_parser, axis=1)
X_test[new_col] = X_test.apply(id_parser, axis=1)
X_train.drop(['PassengerId'], axis=1, inplace=True)
X_test.drop(['PassengerId'], axis=1, inplace=True)

def get_group_dict(group_column):
    groups = group_column.value_counts()
    groups = dict(groups)
    return groups
group_dict = get_group_dict(X_train['GroupNumber'])
group_count = X_train['GroupNumber'].replace(group_dict)
y = []
for i in range(1, 9):
    x = group_count[group_count == i]
    y.append(len(x[Y_train == 1]) / len(x))
sns.lineplot(y=y, x=range(1, 9))
group_dict = get_group_dict(X_train['GroupNumber'])
X_train['GroupNumber'].replace(group_dict, inplace=True)
group_dict = get_group_dict(X_test['GroupNumber'])
X_test['GroupNumber'].replace(group_dict, inplace=True)

def cabin_parser(row):
    s = row['Cabin']
    if s is np.nan:
        return [np.nan] * 3
    (deck, number, side) = s.split('/')
    return [deck, int(number), side == 'S']
new_cols = ['Deck', 'CabinNumber', 'IsSideS']
X_train[new_cols] = X_train.apply(cabin_parser, axis=1, result_type='expand')
X_test[new_cols] = X_test.apply(cabin_parser, axis=1, result_type='expand')
X_train.drop(['Cabin'], axis=1, inplace=True)
X_test.drop(['Cabin'], axis=1, inplace=True)
new_cat_cols = ['Deck', 'IsSideS']
(fig, axes) = plt.subplots(2, 1, figsize=(15, 20))
for (i, col) in enumerate(new_cat_cols):
    axis = axes[i]
    col_data_trans = X_train[col][Y_train == 0].value_counts().to_frame()
    col_data_trans['Target'] = 'Transpoted'
    col_data_not = X_train[col][Y_train == 1].value_counts().to_frame()
    col_data_not['Target'] = 'Not transported'
    col_data = pd.concat([col_data_trans, col_data_not])
    col_data.index = map(str, col_data.index)
    sns.barplot(data=col_data, x=col_data.index, y=col, hue='Target', ax=axis)
    axis.set(ylabel=None, title=col)
group_dict = get_group_dict(X_train['CabinNumber'])
cabin_count = X_train['CabinNumber'].replace(group_dict)
y = []
for i in range(1, 19):
    x = cabin_count[cabin_count == i]
    y.append(len(x[Y_train == 1]) / len(x))
sns.lineplot(y=y, x=range(1, 19))
group_dict = get_group_dict(X_train['CabinNumber'])
X_train['CabinNumber'].replace(group_dict, inplace=True)
group_dict = get_group_dict(X_test['CabinNumber'])
X_test['CabinNumber'].replace(group_dict, inplace=True)
X_train.info()
X_train.describe()
X_train.head()
df = X_train.copy()
df['Transported'] = Y_train.copy()
sns.set(rc={'figure.figsize': (15, 10)})
sns.heatmap(df.corr(), cmap='coolwarm', vmin=-1, vmax=1)

class KFoldTargetEncoder(TransformerMixin):

    def __init__(self, col_names, n_folds=10, smooth=0):
        self.col_names = col_names
        self.n_folds = n_folds
        self.smooth = smooth
        self.replaces = {}

    def fit(self, X, y):
        self.global_mean = y.mean()
        X = X.copy()
        local_means = {}
        for col_name in self.col_names:
            local_means[col_name] = pd.DataFrame(index=pd.unique(X[col_name]))
        kf = KFold(self.n_folds, shuffle=True)
        for (train_ind, _) in kf.split(X):
            X_train = X.iloc[train_ind]
            y_train = y.iloc[train_ind]
            for col_name in self.col_names:
                local_mean = y_train.groupby(X_train[col_name]).mean()
                local_means[col_name] = pd.concat([local_means[col_name], local_mean], axis=1)
        for col_name in self.col_names:
            local_means[col_name].fillna(self.global_mean, inplace=True)
            self.replaces[col_name] = (local_means[col_name].mean(axis=1) * X[col_name].value_counts() + self.smooth * self.global_mean) / (X[col_name].value_counts() + self.smooth)
        return self

    def transform(self, X):
        X = X.copy()
        for col_name in self.col_names:
            new_col_name = col_name + '_encoded'
            X[new_col_name] = X[col_name].map(lambda x: self.replaces[col_name][x] if x in self.replaces[col_name] else self.global_mean)
            X[new_col_name].fillna(self.global_mean, inplace=True)
            X.drop(col_name, axis=1, inplace=True)
        return X
to_target_encoding = ['HomePlanet', 'Destination', 'GroupNumber', 'Deck', 'CabinNumber', 'IsSideS']
encoder = KFoldTargetEncoder(to_target_encoding)
X_train = encoder.fit_transform(X_train, Y_train)
X_test = encoder.transform(X_test)
params = {'learning_rate': 0.15, 'min_child_weight': 0.65, 'gamma': 0.1, 'subsample': 0.75, 'colsample_bytree': 0.85, 'colsample_bylevel': 0.65, 'max_depth': 4, 'n_estimators': 100, 'reg_lambda': 2.25, 'monotone_constraints': '(0, -1)', 'eval_metric': 'logloss', 'use_label_encoder': False}
model = XGBClassifier()
model.set_params(**params)