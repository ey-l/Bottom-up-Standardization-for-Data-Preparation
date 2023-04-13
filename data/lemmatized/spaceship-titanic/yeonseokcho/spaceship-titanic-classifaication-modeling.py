import numpy as np
import pandas as pd
import os
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
_input0['Passenger_Group'] = _input0.PassengerId.str.split('_').str[0]
_input0['Cabin_Deck'] = _input0.Cabin.str.split('/').str[0]
_input0['Cabin_Num'] = _input0.Cabin.str.split('/').str[1]
_input0['Cabin_Side'] = _input0.Cabin.str.split('/').str[2]
_input0['Last_Name'] = _input0.Name.str.split(' ').str[1]
_input0.head(2)
_input0.info()
_input0.describe(include='all')
test = _input0.drop(['PassengerId', 'Cabin', 'Name', 'Passenger_Group', 'Cabin_Num', 'Last_Name'], axis=1)
test
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1['Passenger_Group'] = _input1.PassengerId.str.split('_').str[0]
_input1['Cabin_Deck'] = _input1.Cabin.str.split('/').str[0]
_input1['Cabin_Num'] = _input1.Cabin.str.split('/').str[1]
_input0['Cabin_Side'] = _input1.Cabin.str.split('/').str[2]
_input1['Last_Name'] = _input1.Name.str.split(' ').str[1]
_input1.head(2)
train = _input1.drop(['PassengerId', 'Cabin', 'Name', 'Passenger_Group', 'Cabin_Num', 'Last_Name', 'Transported'], axis=1)
train.head(2)
features_all = pd.concat([test, train])
features_all
features_all.reset_index(drop=True)
target = _input1['Transported']
target
target
target.value_counts()
features_ohe = pd.get_dummies(features_all, drop_first=True)
features_ohe
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_std = scaler.fit_transform(features_ohe)
features_std = pd.DataFrame(features_std, columns=features_ohe.columns)
features_std
features_std.info()
features_std = features_std.fillna(features_std.median())
features_std
features_std.info()
features_std.describe()
corr_rate_threshold = 0.7
cor_matrix = features_std.corr().abs()
cor_matrix
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
upper_tri
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= corr_rate_threshold)]
print(to_drop)
selected_features = features_std.drop(features_std[to_drop], axis=1)
selected_features
features_test = selected_features.loc[0:4276]
features_test
features_train = selected_features.loc[4277:12969]
features_train
from sklearn.model_selection import train_test_split, StratifiedKFold
(X_train, X_val, y_train, y_val) = train_test_split(features_train, target, test_size=0.3, random_state=2301)
(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
from sklearn.dummy import DummyClassifier
bm = DummyClassifier(strategy='stratified', random_state=2302)