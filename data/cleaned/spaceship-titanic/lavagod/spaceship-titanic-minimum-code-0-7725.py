import pandas as pd
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
pas_id = test['PassengerId']
a = pd.DataFrame(train.isnull().sum())
b = pd.DataFrame(test.isnull().sum())
columns_text = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
columns_number = ['Age']
columns_other = ['RoomService', 'Spa', 'VRDeck']
columns_drop = ['FoodCourt', 'ShoppingMall', 'Name', 'PassengerId', 'Cabin']

def fill_mode(df, col):
    df[col].fillna(df[col].mode()[0], inplace=True)

def fill_mean(df, col):
    df[col].fillna(df[col].mean(), inplace=True)

def fill_zero(df, col):
    df[col].fillna(0, inplace=True)
for i in columns_text:
    fill_mode(train, i)
    fill_mode(test, i)
for i in columns_number:
    fill_mean(train, i)
    fill_mean(test, i)
for i in columns_other:
    fill_zero(train, i)
    fill_zero(test, i)
train.drop(columns_drop, axis=1, inplace=True)
test.drop(columns_drop, axis=1, inplace=True)
a = pd.DataFrame(train.isnull().sum())
b = pd.DataFrame(test.isnull().sum())
train = pd.get_dummies(data=train, columns=columns_text, drop_first=True)
test = pd.get_dummies(data=test, columns=columns_text, drop_first=True)
y = train['Transported'].astype('int')
train.drop(['Transported'], axis=1, inplace=True)
X = train
model = LogisticRegression(max_iter=300)