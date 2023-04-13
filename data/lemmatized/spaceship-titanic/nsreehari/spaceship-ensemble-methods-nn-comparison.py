import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
X = _input1.iloc[:, :-1]
y = _input1.iloc[:, -1]
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
ordinal_encoder = OrdinalEncoder()
std_scaler = StandardScaler()
imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

def change(x):
    return ord(x) if x != 'nan' else 1

def preprocess_data(X):
    X['PassengerId'] = X['PassengerId'].str.split('_').str[0]
    freq_col = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']
    mean_col = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for tmp_freq in freq_col:
        X[tmp_freq] = imp_freq.fit_transform(X[tmp_freq].values.reshape(-1, 1))
    for tmp_mean in mean_col:
        X[tmp_mean] = imp_mean.fit_transform(X[tmp_mean].values.reshape(-1, 1))
    ordinal_enc_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'PassengerId']
    for tmp_col in ordinal_enc_columns:
        X[tmp_col] = ordinal_encoder.fit_transform(X.loc[:, tmp_col].values.reshape(-1, 1))
    X['PassengerId'] = X['PassengerId'] / X['PassengerId'].max()
    cab_0 = X['Cabin'].str.split('/').str[0]
    cab_1 = X['Cabin'].str.split('/').str[1]
    cab_2 = X['Cabin'].str.split('/').str[2]
    cab_0 = cab_0.astype('str').map(change)
    cab_1 = cab_1.fillna(1)
    cab_1[cab_1.astype('int') == 0] = 0.5
    cab_1 = cab_1.astype('float') / cab_1.astype('float').max()
    cab_2 = cab_2.astype('str').map(change)
    X['Cabin'] = cab_0 * cab_1 * cab_2
    std_scale_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin']
    for tmp_col in std_scale_columns:
        X[tmp_col] = std_scaler.fit_transform(X.loc[:, tmp_col].values.reshape(-1, 1))
    X = X.drop(['Name'], axis=1)
    return X
X = _input1.iloc[:, :-1]
X
X = preprocess_data(X)
X
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, shuffle=True, test_size=0.2)
from sklearn.metrics import precision_score, recall_score, f1_score

def score(y_test, y_pred):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'Precision => {precision}\nRecall => {recall}\nF1 score => {f1}')
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()