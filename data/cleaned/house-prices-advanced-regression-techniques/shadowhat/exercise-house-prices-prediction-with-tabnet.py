import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from pytorch_tabnet.tab_model import TabNetRegressor
import random
BATCH_SIZE = 30
EPOCHS = 1600
TOTAL_EPOCHS = 3
LEARNING_RATE = 0.001
VALIDATION_RATIO = 0.07
Y_COL_NAME = 'SalePrice'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
print(DEVICE)
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_df

def check_null_index(df):
    null_check_df = df.isnull().any()
    non_null_index_list = list(null_check_df[null_check_df == False].index)
    null_index_list = list(null_check_df[null_check_df == True].index)
    print(non_null_index_list)
    print(null_index_list)
    return null_index_list
train_nullable_col = check_null_index(train_df)
test_nullable_col = check_null_index(test_df)

def checkHowMuchNullInColumn(index_list, df):
    for index_e in index_list:
        print('index : {}'.format(index_e))
        len_of_null_in_column = len(df[train_df[index_e].isnull()])
        print('null element : {}'.format(len_of_null_in_column))
    return
checkHowMuchNullInColumn(train_nullable_col, train_df)
checkHowMuchNullInColumn(test_nullable_col, test_df)

def getNonNumericColumn(df):
    categorical_dtypes_index_list = df.select_dtypes(include='object').columns.tolist()
    return categorical_dtypes_index_list

def getNumericColumn(df):
    categorical_dtypes_index_list = df.select_dtypes(include='number').columns.tolist()
    return categorical_dtypes_index_list
test_num_index_list = getNumericColumn(test_df)
print(test_num_index_list)
train_df = train_df.drop(columns=['Id'])
test_df = test_df.drop(columns=['Id'])
train_cat_index_list = getNonNumericColumn(train_df)
test_cat_index_list = getNonNumericColumn(test_df)
train_num_index_list = getNumericColumn(train_df)
test_num_index_list = getNumericColumn(test_df)
train_df[train_num_index_list] = train_df[train_num_index_list].fillna(train_df[train_num_index_list].mean())
test_df[test_num_index_list] = test_df[test_num_index_list].fillna(train_df[test_num_index_list].mean())
train_df[train_cat_index_list] = train_df[train_cat_index_list].fillna('NULL')
test_df[test_cat_index_list] = test_df[test_cat_index_list].fillna('NULL')
train_df[train_cat_index_list] = train_df[train_cat_index_list].astype('category')
test_df[test_cat_index_list] = test_df[test_cat_index_list].astype('category')
train_df.head(8)
for train_cat_index_element in train_cat_index_list:
    temp_df = pd.get_dummies(train_df[train_cat_index_element], prefix=train_cat_index_element, dummy_na=True)
    train_df = train_df.drop(columns=[train_cat_index_element])
    train_df = pd.concat([train_df, temp_df], axis=1)
for test_cat_index_element in test_cat_index_list:
    temp_df = pd.get_dummies(test_df[test_cat_index_element], prefix=test_cat_index_element, dummy_na=True)
    test_df = test_df.drop(columns=[test_cat_index_element])
    test_df = pd.concat([test_df, temp_df], axis=1)
train_column_list = train_df.columns.tolist()
test_column_list = test_df.columns.tolist()
print(set(train_column_list) - set(test_column_list))
print(set(test_column_list) - set(train_column_list))
all_column_list = list(set(train_column_list).union(set(test_column_list)))
all_column_list
train_df
test_df
train_df = train_df.reindex(columns=all_column_list).fillna(0)
test_df = test_df.reindex(columns=all_column_list).fillna(0)
null_check_df = train_df.isnull().any()
null_check_df[null_check_df == True]
null_check_df = test_df.isnull().any()
null_check_df[null_check_df == True]
train_df.head(10)
test_df.head(10)
test_df = test_df.drop(columns=[Y_COL_NAME])

class House_price_train_dataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.x_data = self.df.drop(columns=[Y_COL_NAME]).to_numpy()
        self.y_data = self.df[Y_COL_NAME].to_numpy().reshape(-1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_data = self.x_data[idx]
        y_data = self.y_data[idx]
        return (x_data, y_data)
whole_train_dataset = House_price_train_dataset(train_df)
whole_train_dataset[1]

def fit(whole_train_dataset):
    cost_list = []
    model_list = []
    for _ in range(0, TOTAL_EPOCHS):
        len_whole_train_dataset = len(whole_train_dataset)
        len_validation_dataset = int(VALIDATION_RATIO * len_whole_train_dataset)
        len_train_dataset = len_whole_train_dataset - len_validation_dataset
        (train_dataset, validation_dataset) = random_split(whole_train_dataset, [len_train_dataset, len_validation_dataset])
        model = TabNetRegressor(seed=1)
        x_train = np.array([row_data[0] for row_data in train_dataset])
        y_train = np.array([row_data[1] for row_data in train_dataset])
        x_valid = np.array([row_data[0] for row_data in validation_dataset])
        y_valid = np.array([row_data[1] for row_data in validation_dataset])