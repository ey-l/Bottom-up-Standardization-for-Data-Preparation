import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def list_extract(big_list, small_list):
    return list(set(big_list) - set(small_list))

def type_extract(data, type_name, True_or_False=True):
    arblist = []
    for i in data.columns:
        if data[i].dtype == type_name:
            arblist.append(i)
    if True_or_False == True:
        return arblist
    else:
        return list_extract(data.columns, arblist)
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train['Is_train'] = True
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test['Is_train'] = False
data = pd.concat([train.drop('Transported', axis=1), test], axis=0, ignore_index=True)
obj_col = type_extract(data, 'object')
num_col = type_extract(data, 'object', False)
obj_data = data[obj_col]
num_data = data[num_col]
obj_data['Front_Id'] = obj_data['PassengerId'].str.split('_').str[0]
obj_data['Rear_Id'] = obj_data['PassengerId'].str.split('_').str[1]
obj_data = obj_data.drop('PassengerId', axis=1)
obj_data['Deck'] = obj_data['Cabin'].str.split('/').str[0]
obj_data['Num'] = obj_data['Cabin'].str.split('/').str[1]
obj_data['Side'] = obj_data['Cabin'].str.split('/').str[2]
obj_data = obj_data.drop('Cabin', axis=1)
obj_data['First_Name'] = obj_data['Name'].str.split().str[0]
obj_data['Last_Name'] = obj_data['Name'].str.split().str[1]
obj_data = obj_data.drop('Name', axis=1)
num_data = pd.concat([num_data, obj_data[['Front_Id', 'Rear_Id', 'Num']].astype('float')], axis=1)
obj_data.drop(['Front_Id', 'Rear_Id', 'Num'], axis=1, inplace=True)
obj_data = obj_data.fillna(obj_data.mode().loc[0])
obj_data_dum = pd.get_dummies(data=obj_data, columns=type_extract(obj_data, 'object'), drop_first=True)
num_data = num_data.fillna(num_data.median())
data = pd.concat([obj_data_dum, num_data], axis=1)
data
data_train_X = data[data['Is_train'] == True].drop('Is_train', axis=1)
data_train_y = train['Transported']
data_test_X = data[data['Is_train'] == False].drop('Is_train', axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(data_train_X, data_train_y, test_size=0.2, random_state=100)
bp = {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500, 'subsample': 1}
from xgboost import XGBClassifier