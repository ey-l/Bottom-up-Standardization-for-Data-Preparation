import numpy as np
import pandas as pd
import os
test_raw = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_raw
test_raw['Passenger_Group'] = test_raw.PassengerId.str.split('_').str[0]
test_raw['Cabin_Deck'] = test_raw.Cabin.str.split('/').str[0]
test_raw['Cabin_Num'] = test_raw.Cabin.str.split('/').str[1]
test_raw['Cabin_Side'] = test_raw.Cabin.str.split('/').str[2]
test_raw['Last_Name'] = test_raw.Name.str.split(' ').str[1]
test_raw.head(2)
test_raw.info()
test_raw.describe(include='all')
test = test_raw.drop(['PassengerId', 'Cabin', 'Name', 'Passenger_Group', 'Cabin_Num', 'Last_Name'], axis=1)
test
train_raw = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_raw
train_raw['Passenger_Group'] = train_raw.PassengerId.str.split('_').str[0]
train_raw['Cabin_Deck'] = train_raw.Cabin.str.split('/').str[0]
train_raw['Cabin_Num'] = train_raw.Cabin.str.split('/').str[1]
test_raw['Cabin_Side'] = train_raw.Cabin.str.split('/').str[2]
train_raw['Last_Name'] = train_raw.Name.str.split(' ').str[1]
train_raw.head(2)
train = train_raw.drop(['PassengerId', 'Cabin', 'Name', 'Passenger_Group', 'Cabin_Num', 'Last_Name', 'Transported'], axis=1)
train.head(2)
features_all = pd.concat([test, train])
features_all
features_all.reset_index(drop=True)
target = train_raw['Transported']
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