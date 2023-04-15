from sklearn import linear_model, preprocessing, impute, model_selection, metrics
from scipy.stats import boxcox
import pandas as pd
import numpy as np
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
sns.set_style('ticks')
sns.despine()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything()
DATA_DIR = 'data/input/spaceship-titanic'

def filepath(filename):
    return os.path.join(DATA_DIR, filename)
train_df = pd.read_csv(filepath('train.csv'), index_col='PassengerId')
test_df = pd.read_csv(filepath('test.csv'), index_col='PassengerId')
train_df['PassengerId'] = train_df.index
test_df['PassengerId'] = test_df.index
(len(train_df), len(test_df))
expenditure_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

def from_passengerId(df):
    split_id = df['PassengerId'].str.split('_', expand=True)
    df['GroupId'] = split_id[0]
    df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
    df['Alone'] = df['GroupSize'] == 1
    return df
train_df = from_passengerId(train_df)
test_df = from_passengerId(test_df)
train_df.head()

def missing_value_features(df, columns, expenditure_columns):
    for column in columns:
        df[f'{column}_missing'] = df[column].isna()
    df['TotalExpense_missing'] = df[expenditure_columns].sum(axis=1, skipna=False).isna()
    return df
columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Cabin', 'VIP']
train_df = missing_value_features(train_df, columns, expenditure_columns)
test_df = missing_value_features(test_df, columns, expenditure_columns)
train_df.head()

def from_expenditure_features(df, expenditure_columns):
    df['TotalExpense'] = df[expenditure_columns].sum(axis=1)
    return df
train_df = from_expenditure_features(train_df, expenditure_columns)
test_df = from_expenditure_features(test_df, expenditure_columns)
train_df.head()

def from_cabin(df):
    df[['CabinDeck', 'CabinNum', 'CabinSide']] = df['Cabin'].str.split('/', expand=True)
    return df

def simple_mode_replacement(df, columns):
    df[columns] = df[columns].fillna(df[columns].mode().iloc[0])
    return df
columns = ['HomePlanet', 'CryoSleep', 'Destination']
train_df = simple_mode_replacement(train_df, columns)
test_df = simple_mode_replacement(test_df, columns)
train_df[columns].isna().any()
train_df.head()

def group_mode_replacement(df, groupby, column):
    temp = df.groupby(groupby).filter(lambda x: x[column].notna().any())
    func = lambda x: x.fillna(x.mode().iloc[0]) if x.isna().any() else x
    temp[column] = temp.groupby(groupby)[column].transform(func)
    df.loc[temp.index, column] = temp[column]
    return df
train_df = group_mode_replacement(train_df, groupby='GroupId', column='Cabin')
test_df = group_mode_replacement(test_df, groupby='GroupId', column='Cabin')
train_df.head()
train_df['Cabin'].isna().sum()
test_df['Cabin'].isna().sum()
train_df = group_mode_replacement(train_df, groupby=['HomePlanet', 'Destination'], column='Cabin')
test_df = group_mode_replacement(test_df, groupby=['HomePlanet', 'Destination'], column='Cabin')
train_df['Cabin'].isna().sum()
test_df['Cabin'].isna().sum()
train_df = from_cabin(train_df)
test_df = from_cabin(test_df)
train_df.head()
columns = ['Cabin', 'CabinDeck', 'CabinNum', 'CabinSide']
train_df[columns].isna().any()
test_df[columns].isna().any()
(train_df['VIP'].isna().sum(), test_df['VIP'].isna().sum())

def impute_vip_for_no_spend(df):
    df.loc[df['VIP'].isna() & (df['TotalExpense'] == 0.0) & ~df['CryoSleep'], 'VIP'] = False
    return df

def impute_vip_for_children(df):
    df.loc[df['VIP'].isna() & (df['Age'] <= 12), 'VIP'] = False
    return df

def impute_vip_for_earthlings(df):
    df.loc[df['VIP'].isna() & (df['HomePlanet'] == 'Earth'), 'VIP'] = False
    return df

def impute_vip_for_martians(df):
    df.loc[df['VIP'].isna() & (df['Age'] >= 18) & ~df['CryoSleep'] & (df['Destination'] != '55 Cancri e'), 'VIP'] = True
    return df

def impute_vip(df):
    df = impute_vip_for_no_spend(df)
    df = impute_vip_for_children(df)
    df = impute_vip_for_earthlings(df)
    df = impute_vip_for_martians(df)
    return df
train_df = impute_vip(train_df)
test_df = impute_vip(test_df)
(train_df['VIP'].isna().sum(), test_df['VIP'].isna().sum())

def impute_vip_by_prob(df):
    probs = df['VIP'].value_counts() / df['VIP'].notna().sum()
    values = np.random.choice([False, True], size=df['VIP'].isna().sum(), p=probs)
    df.loc[df['VIP'].isna(), 'VIP'] = values
    df['VIP'] = df['VIP'].astype(bool)
    return df
train_df = impute_vip_by_prob(train_df)
test_df = impute_vip_by_prob(test_df)
(train_df['VIP'].isna().sum(), test_df['VIP'].isna().sum())
drop = ['PassengerId', 'Cabin', 'Name']
train_df = train_df.drop(drop, axis=1)
test_df = test_df.drop(drop, axis=1)
train_df.head()
train_df.isna().any()

def concat_train_test(train, test, has_labels=False):
    transported = None
    if has_labels is True:
        transported = train['Transported'].copy()
        train = train.drop('Transported', axis=1)
    train_index = train.index
    test_index = test.index
    df = pd.concat([train, test])
    return (df, train_index, test_index, transported)

def split_train_test(df, train_index, test_index, transported=None):
    train_df = df.loc[train_index, :]
    if transported is not None:
        train_df['Transported'] = transported
    test_df = df.loc[test_index, :]
    return (train_df, test_df)
(df, train_idx, test_idx, transported) = concat_train_test(train_df, test_df, has_labels=True)
df.head()

def bool2int(df):
    columns = [column for column in df.columns if df[column].dtype.name == 'bool']
    df[columns] = df[columns].astype(int)
    return df
df = bool2int(df)
df.head()
df['CabinSide'] = df['CabinSide'].map({'S': 0, 'P': 1})
to_be_encoded = ['HomePlanet', 'Destination', 'GroupSize', 'CabinDeck']
df = pd.get_dummies(df, columns=to_be_encoded)
df.head()
df.columns
(train_df, test_df) = split_train_test(df, train_idx, test_idx, transported=transported)
train_df.head()
test_df.head()

def impute_missing_using_knn(df, numeric_cols, has_labels=False):
    x = df
    if has_labels is True:
        transported = df['Transported']
        x = df.drop('Transported', axis=1)
    scaler = preprocessing.StandardScaler()
    x[numeric_cols] = scaler.fit_transform(x[numeric_cols])
    imputer = impute.KNNImputer(n_neighbors=5, weights='distance')
    x = imputer.fit_transform(x)
    if has_labels is True:
        x = np.hstack((x, transported.values.reshape(-1, 1)))
    return pd.DataFrame(x, columns=df.columns, index=df.index)
train_cabin_num = train_df['CabinNum']
train_group_id = train_df['GroupId']
test_cabin_num = test_df['CabinNum']
test_group_id = test_df['GroupId']
to_drop = ['GroupId', 'CabinNum']
numeric_cols = ['Age', 'TotalExpense'] + expenditure_columns
train_df = impute_missing_using_knn(train_df.drop(to_drop, axis=1), numeric_cols, has_labels=True)
test_df = impute_missing_using_knn(test_df.drop(to_drop, axis=1), numeric_cols)
train_df.head()
test_df.head()
train_df.isna().any()
test_df.head()
train_df = train_df.reset_index()
train_df['kfold'] = -1
kf = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
for (idx, (_, val_idx)) in enumerate(kf.split(train_df)):
    train_df.loc[val_idx, 'kfold'] = idx
train_df = train_df.set_index('PassengerId')
train_df.head()
len(train_df[train_df['kfold'] != 0])



def train(df):
    df['preds'] = pd.NA
    drop = ['Transported', 'preds', 'kfold']
    for fold in range(5):
        train = df[df['kfold'] != fold]
        y_train = train['Transported'].values
        X_train = train.drop(drop, axis=1).values
        val = df[df['kfold'] == fold]
        y_val = val['Transported'].values
        X_val = val.drop(drop, axis=1).values
        model = linear_model.LogisticRegression(max_iter=1000)