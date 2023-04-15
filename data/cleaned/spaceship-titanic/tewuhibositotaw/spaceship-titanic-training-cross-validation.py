import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_df.info()
test_df.info()


train_df.describe()
test_df.describe()
msno.matrix(train_df)
msno.matrix(test_df)
train_df[['CryoSleep', 'VIP', 'Transported']] = (train_df[['CryoSleep', 'VIP', 'Transported']] == True).astype(int)
test_df[['CryoSleep', 'VIP']] = (test_df[['CryoSleep', 'VIP']] == True).astype(int)
data_df = pd.concat([train_df, test_df], axis=0)
data_df
msno.matrix(data_df)
data_df['Age'].fillna(data_df['Age'].mean(), inplace=True)
data_df['HomePlanet'].fillna('Europa', inplace=True)
data_df['CryoSleep'].fillna('False', inplace=True)
data_df['Cabin'].fillna('X0000', inplace=True)
data_df['Destination'].fillna('55 Cancri e', inplace=True)
data_df['VIP'].fillna('False', inplace=True)
data_df['RoomService'].fillna(data_df['RoomService'].mean(), inplace=True)
data_df['FoodCourt'].fillna(data_df['FoodCourt'].mean(), inplace=True)
data_df['ShoppingMall'].fillna(data_df['ShoppingMall'].mean(), inplace=True)
data_df['Spa'].fillna(data_df['Spa'].mean(), inplace=True)
data_df['VRDeck'].fillna(data_df['VRDeck'].mean(), inplace=True)
data_df['Name'].fillna('Name not known', inplace=True)
msno.matrix(data_df)
train = data_df[0:len(train_df)]
test = data_df[len(train_df):]
train.info()
data_oh = pd.get_dummies(data_df, columns=['HomePlanet', 'Destination'])
data_df
data_oh
data_oh.drop(['PassengerId', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'], axis=1, inplace=True)
data_oh
from sklearn.preprocessing import StandardScaler
num_cols = ['Age']
data_oh.describe()
data_std = data_oh.copy()
scaler = StandardScaler()
data_std[num_cols] = scaler.fit_transform(data_std[num_cols])
data_oh.describe()
data_std
data_std.describe()
train = data_std[0:len(train_df)]
test = data_std[len(train_df):]
y = train['Transported']
X = train.drop('Transported', axis=1)
X_test = test.drop('Transported', axis=1)
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
scores = cross_val_score(model, X, y, scoring='accuracy', cv=5, verbose=20)
print(scores)
print(np.mean(scores), np.std(scores))
cv_result = cross_validate(model, X, y, scoring='accuracy', cv=5, verbose=20, return_train_score=True, return_estimator=True)
cv_result
cv_result['estimator']
cv_result['estimator'][0].get_params()
result01 = cv_result['estimator'][0].predict(X_test)
result01
result02 = cv_result['estimator'][1].predict(X_test)
print(result02)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
kfold = KFold(n_splits=5, shuffle=True, random_state=1026)
scores = []
for (tr_idx, val_idx) in kfold.split(X):
    (X_tr, X_val) = (X.iloc[tr_idx], X.iloc[val_idx])
    (y_tr, y_val) = (y.iloc[tr_idx], y.iloc[val_idx])
    model = LogisticRegression()