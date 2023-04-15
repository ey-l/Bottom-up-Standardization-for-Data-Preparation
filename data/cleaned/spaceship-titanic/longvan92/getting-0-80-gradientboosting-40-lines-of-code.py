import os
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head()

def data_edit(df):
    df['Desk'] = [cabin[0] if type(cabin) == str else cabin for cabin in df['Cabin']]
    df['Side'] = [cabin[-1] if type(cabin) == str else cabin for cabin in df['Cabin']]
    df = df.drop(columns=['PassengerId', 'Cabin', 'Name'])
    df = pd.get_dummies(df, drop_first=True)
    return df

def data_scaler(df_train, df_test):
    most_frequent = df_train.mode().loc[0, :]
    df_train = df_train.fillna(value=most_frequent)
    df_test = df_test.fillna(value=most_frequent)
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    std = preprocessing.StandardScaler()
    df_train[num_cols] = std.fit_transform(df_train[num_cols])
    df_test[num_cols] = std.transform(df_test[num_cols])
    return (df_train, df_test)
df_train = data_edit(df_train)
df_test = data_edit(df_test)
(df_train, df_test) = data_scaler(df_train, df_test)
df_train.head()
X_train = df_train.drop('Transported', axis=1)
y_train = df_train['Transported']
X_test = df_test
clf = GradientBoostingClassifier()