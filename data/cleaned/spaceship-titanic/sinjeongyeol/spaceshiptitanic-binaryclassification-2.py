import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train.info()
feats_cat = []
feats_num = []
for col in train.columns:
    if col in ['PassengerId', 'Transported']:
        continue
    if train[col].dtype == 'object':
        feats_cat += [col]
    else:
        feats_num += [col]
(feats_cat, feats_num)
train_len = len(train)
dataset = pd.concat([train, test], axis=0).reset_index(drop=True)
(dataset.shape, train.shape, test.shape)
dataset['Transported'] = pd.to_numeric(dataset['Transported'])
train['PassengerId']
dataset['group_id'] = 0
dataset['group_num'] = 0
for i in range(len(dataset)):
    (group_id, id_at_group) = map(int, dataset.iloc[i]['PassengerId'].split('_'))
    dataset.loc[i, 'group_id'] = group_id
    dataset.loc[i, 'group_num'] = id_at_group
dataset = pd.merge(dataset, dataset[['group_id', 'PassengerId']].groupby('group_id').count().reset_index().rename(columns={'PassengerId': 'group_cnt'}), how='left', on='group_id')
dataset = dataset.drop('group_id', axis=1)
dataset.head()
feats_num += ['group_num', 'group_cnt']
import seaborn as sns
corr = dataset[feats_num].corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
import matplotlib.pyplot as plt
(fig, ax) = plt.subplots(1, 2, figsize=(10, 6))
dataset[dataset['Transported'].notnull()][['group_cnt', 'Transported']].groupby('group_cnt').mean().plot.bar(ax=ax[0])
sns.countplot(x='group_cnt', data=dataset[dataset['Transported'].notnull()], hue='Transported', ax=ax[1])
(fig, ax) = plt.subplots(1, 2, figsize=(10, 6))
dataset[dataset['Transported'].notnull()][['group_num', 'Transported']].groupby('group_num').mean().plot.bar(ax=ax[0])
sns.countplot(x='group_num', data=dataset[dataset['Transported'].notnull()], hue='Transported', ax=ax[1])
(feats_num, feats_cat)
dataset['Cabin']
dataset['Cabin'].isnull().mean()
dataset['Cabin_deck'] = np.NaN
dataset['Cabin_num'] = np.NaN
dataset['Cabin_side'] = np.NaN
for i in range(len(dataset)):
    if pd.isna(dataset.iloc[i]['Cabin']):
        continue
    (deck, num, side) = dataset.iloc[i]['Cabin'].split('/')
    dataset.loc[i, 'Cabin_deck'] = deck
    dataset.loc[i, 'Cabin_num'] = int(num)
    dataset.loc[i, 'Cabin_side'] = side
dataset.head()
dataset['Cabin_deck'].value_counts()
(fig, ax) = plt.subplots(1, 2, figsize=(10, 6))
dataset[dataset['Transported'].notnull()][['Cabin_deck', 'Transported']].groupby('Cabin_deck').mean().plot.bar(ax=ax[0])
sns.countplot(x='Cabin_deck', data=dataset[dataset['Transported'].notnull()], hue='Transported', ax=ax[1])
dataset['Cabin_num'].value_counts()
sns.kdeplot(x='Cabin_num', data=dataset)
sns.kdeplot(x='Cabin_num', data=dataset[dataset['Transported'].notnull()], hue='Transported')
dataset['Cabin_side'].value_counts()
(fig, ax) = plt.subplots(1, 2, figsize=(10, 6))
dataset[dataset['Transported'].notnull()][['Cabin_side', 'Transported']].groupby('Cabin_side').mean().plot.bar(ax=ax[0])
sns.countplot(x='Cabin_side', data=dataset[dataset['Transported'].notnull()], hue='Transported', ax=ax[1])
dataset = dataset.drop('Cabin', axis=1)
dataset.head()
if 'Cabin' in feats_cat:
    feats_cat.remove('Cabin')
feats_cat.append('Cabin_deck')
feats_cat.append('Cabin_side')
feats_num.append('Cabin_num')
(feats_cat, feats_num)
train_ = dataset[dataset['Transported'].notnull()]
test_ = dataset[dataset['Transported'].isnull()]
dataset.info()
trn_series = train_['HomePlanet']
tst_series = test_['HomePlanet']
target = train_['Transported']
temp = pd.concat([trn_series, target], axis=1)
cumsum = temp.groupby(trn_series.name)[target.name].cumsum() - target
cumcnt = temp.groupby(trn_series.name).cumcount() + 1
trn_series_new = cumsum / cumcnt
temp['trn_series_new'] = trn_series_new
sns.kdeplot(x='trn_series_new', data=temp, hue=trn_series.name)

def smoothing(n_rows, target_mean, global_mean, alpha):
    return (target_mean * n_rows + global_mean * alpha) / (n_rows + alpha)
global_mean = target.mean()
alpha = 0.7
mean_cnt = temp.groupby(trn_series.name)[target.name].agg(['mean', 'count'])
mean_cnt['mean_smoothing'] = mean_cnt.apply(lambda x: smoothing(x['count'], x['mean'], global_mean, alpha), axis=1)
mean_cnt.drop(['mean', 'count'], axis=1, inplace=True)
mean_cnt
tst_df = pd.merge(tst_series.to_frame(tst_series.name), mean_cnt.reset_index(), on=tst_series.name, how='left')
tst_df.index = tst_series.index
tst_df
tst_df[tst_df[tst_series.name].isnull()]

def smoothing(n_rows, target_mean, global_mean, alpha):
    return (target_mean * n_rows + global_mean * alpha) / (n_rows + alpha)

def mean_encode(trn_series, val_series, tst_series, target, alpha):
    temp = pd.concat([trn_series, target], axis=1)
    mean_cnt = temp.groupby(trn_series.name)[target.name].agg(['mean', 'count'])
    global_mean = target.mean()
    mean_cnt['mean_smoothing'] = mean_cnt.apply(lambda x: smoothing(x['count'], x['mean'], global_mean, alpha), axis=1)
    mean_cnt.drop(['mean', 'count'], axis=1, inplace=True)
    cumsum = temp.groupby(trn_series.name)[target.name].cumsum() - target
    cumcnt = temp.groupby(trn_series.name).cumcount() + 1
    trn_series_new = cumsum / cumcnt
    val_df = pd.merge(val_series.to_frame(val_series.name), mean_cnt.reset_index(), on=val_series.name, how='left')
    val_df.index = val_series.index
    tst_df = pd.merge(tst_series.to_frame(tst_series.name), mean_cnt.reset_index(), on=tst_series.name, how='left')
    tst_df.index = tst_series.index
    return (trn_series_new, val_df['mean_smoothing'], tst_df['mean_smoothing'])
dataset.head()
feats_cat
set(dataset.columns) - set(feats_cat + feats_num)
feats_bin = []
for col in feats_cat:
    val_arr = np.array(dataset[col].unique())
    val_arr = val_arr[~pd.isnull(val_arr)]
    if len(val_arr) == 2:
        feats_bin += [col]
        feats_cat.remove(col)
feats_bin
feats_cat
feats_num
feats_cat.remove('Name')
dataset.drop('Name', axis=1, inplace=True)
dataset.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cry = dataset['CryoSleep'].copy()
cry[cry.notnull()] = le.fit_transform(cry[cry.notnull()])
cry.unique()
dataset.head()
dataset.drop('PassengerId', axis=1, inplace=True)
train_val = dataset[dataset['Transported'].notnull()].copy()
test = dataset[dataset['Transported'].isnull()].copy()
test.drop('Transported', axis=1, inplace=True)
y = train_val['Transported']
X = train_val.drop('Transported', axis=1)
X = X.fillna(-1)
test = test.fillna(-1)
X[feats_bin] = X[feats_bin].astype('str')
test[feats_bin] = test[feats_bin].astype('str')
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

def acc(pred, target):
    return (pred == target).mean()

def get_oof(name, model):
    val_predict = np.zeros((len(y),))
    test_predict = np.zeros((len(test), 5))
    print('TRAIN ', name)
    for (cv, (train_idx, val_idx)) in enumerate(kfold.split(X)):
        X_test = test.copy()
        (X_train, y_train) = (X.iloc[train_idx].copy(), y.iloc[train_idx].copy())
        (X_val, y_val) = (X.iloc[val_idx].copy(), y.iloc[val_idx].copy())
        for col in feats_bin:
            X_train_bin = X_train[col].copy()
            le = LabelEncoder()