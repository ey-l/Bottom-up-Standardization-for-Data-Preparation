import pandas as pd
from sklearn.linear_model import LogisticRegression
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
pas_id = _input0['PassengerId']
a = pd.DataFrame(_input1.isnull().sum())
b = pd.DataFrame(_input0.isnull().sum())
columns_text = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
columns_number = ['Age']
columns_other = ['RoomService', 'Spa', 'VRDeck']
columns_drop = ['FoodCourt', 'ShoppingMall', 'Name', 'PassengerId', 'Cabin']

def fill_mode(df, col):
    df[col] = df[col].fillna(df[col].mode()[0], inplace=False)

def fill_mean(df, col):
    df[col] = df[col].fillna(df[col].mean(), inplace=False)

def fill_zero(df, col):
    df[col] = df[col].fillna(0, inplace=False)
for i in columns_text:
    fill_mode(_input1, i)
    fill_mode(_input0, i)
for i in columns_number:
    fill_mean(_input1, i)
    fill_mean(_input0, i)
for i in columns_other:
    fill_zero(_input1, i)
    fill_zero(_input0, i)
_input1 = _input1.drop(columns_drop, axis=1, inplace=False)
_input0 = _input0.drop(columns_drop, axis=1, inplace=False)
a = pd.DataFrame(_input1.isnull().sum())
b = pd.DataFrame(_input0.isnull().sum())
_input1 = pd.get_dummies(data=_input1, columns=columns_text, drop_first=True)
_input0 = pd.get_dummies(data=_input0, columns=columns_text, drop_first=True)
y = _input1['Transported'].astype('int')
_input1 = _input1.drop(['Transported'], axis=1, inplace=False)
X = _input1
model = LogisticRegression(max_iter=300)