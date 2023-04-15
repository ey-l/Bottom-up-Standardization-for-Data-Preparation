import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print(pd.__version__)
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
print(sklearn.__version__)
print(matplotlib.__version__)
train_file = 'data/input/spaceship-titanic/train.csv'
test_file = 'data/input/spaceship-titanic/test.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
print(train_data.head())
train_data.info()
null_cols = train_data.columns[train_data.isnull().any()].tolist()
null_count_cols = train_data[null_cols].isnull().sum()
print('missing values in train data:')
print(null_cols)
plt.figure(figsize=(20, 10))
plt.bar(null_cols, null_count_cols)

num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
print('numeric columns:', num_cols)
for col in num_cols:
    plt.figure(figsize=(15, 7))
    plt.title(col)
    plt.hist(train_data[col], bins=100)

corr_mat = train_data[train_data.columns].corr(method='spearman')
corr = corr_mat['Transported'].drop('Transported').sort_values(ascending=False)
print(corr)

def fill_null_according_PassengerId(data: pd.DataFrame):
    data['Group'] = data['PassengerId'].apply(lambda x: x.split('_')[0])
    fill_cols = ['HomePlanet', 'Destination', 'Cabin']
    for col in fill_cols:
        groupby_col = data.groupby('Group', dropna=False)[col]
        for group in groupby_col.groups:
            if groupby_col.get_group(group).isnull().any():
                if groupby_col.get_group(group).any():
                    data.loc[data['Group'] == group, col] = groupby_col.get_group(group).value_counts().idxmax()
                else:
                    data.loc[data['Group'] == group, col] = data[col].mode()[0]

def split_cabin_info(data: pd.DataFrame):
    data['Deck'] = data['Cabin'].apply(lambda x: x.split('/')[0])
    data['Side'] = data['Cabin'].apply(lambda x: x.split('/')[2])

def fill_other_null(data: pd.DataFrame):
    fill_mean_cols = ['Age']
    for col in fill_mean_cols:
        data[col] = data[col].fillna(data[col].mean())
    fill_zero_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in fill_zero_cols:
        data[col] = data[col].fillna(0)
    fill_false_cols = ['CryoSleep', 'VIP']
    for col in fill_false_cols:
        data[col] = data[col].fillna(False)

def del_cols(data: pd.DataFrame):
    del_cols = ['PassengerId', 'Name', 'Group', 'Cabin']
    data.drop(del_cols, axis=1, inplace=True)

def preprocess(data: pd.DataFrame):
    new_data = data.copy()
    fill_null_according_PassengerId(new_data)
    split_cabin_info(new_data)
    fill_other_null(new_data)
    del_cols(new_data)
    for col in new_data.columns:
        if new_data[col].dtype == 'object' or new_data[col].dtype == 'bool':
            encoder = LabelEncoder()