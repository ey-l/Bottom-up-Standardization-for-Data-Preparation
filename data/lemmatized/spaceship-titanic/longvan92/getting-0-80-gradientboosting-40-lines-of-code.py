import os
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()

def data_edit(df):
    df['Desk'] = [cabin[0] if type(cabin) == str else cabin for cabin in df['Cabin']]
    df['Side'] = [cabin[-1] if type(cabin) == str else cabin for cabin in df['Cabin']]
    df = df.drop(columns=['PassengerId', 'Cabin', 'Name'])
    df = pd.get_dummies(df, drop_first=True)
    return df

def data_scaler(df_train, df_test):
    most_frequent = _input1.mode().loc[0, :]
    _input1 = _input1.fillna(value=most_frequent)
    _input0 = _input0.fillna(value=most_frequent)
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    std = preprocessing.StandardScaler()
    _input1[num_cols] = std.fit_transform(_input1[num_cols])
    _input0[num_cols] = std.transform(_input0[num_cols])
    return (_input1, _input0)
_input1 = data_edit(_input1)
_input0 = data_edit(_input0)
(_input1, _input0) = data_scaler(_input1, _input0)
_input1.head()
X_train = _input1.drop('Transported', axis=1)
y_train = _input1['Transported']
X_test = _input0
clf = GradientBoostingClassifier()