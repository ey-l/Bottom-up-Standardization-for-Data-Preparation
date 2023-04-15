import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import plotly.express as px
import numpy as np
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_df
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df
train_df.info()
test_df.info()
train_df['PassengerId_g'] = train_df['PassengerId'].str[0:4]
train_df['PassengerId_g'] = train_df['PassengerId_g'].astype(int)
test_df['PassengerId_g'] = test_df['PassengerId'].str[0:4]
test_df['PassengerId_g'] = test_df['PassengerId_g'].astype(int)
train_df['PassengerId_p'] = train_df['PassengerId'].str[-2:]
train_df['PassengerId_p'] = train_df['PassengerId_p'].astype(int)
test_df['PassengerId_p'] = test_df['PassengerId'].str[-2:]
test_df['PassengerId_p'] = test_df['PassengerId_p'].astype(int)
train_df['Cabin_deck'] = train_df['Cabin'].str[0:1]
train_df['Cabin_deck'] = train_df['Cabin_deck'].astype(str)
test_df['Cabin_deck'] = test_df['Cabin'].str[0:1]
test_df['Cabin_deck'] = test_df['Cabin_deck'].astype(str)
train_df['Cabin_num'] = train_df['Cabin'].str[2:-2]
train_df['Cabin_num'] = train_df['Cabin_num'].astype(float)
test_df['Cabin_num'] = test_df['Cabin'].str[2:-2]
test_df['Cabin_num'] = test_df['Cabin_num'].astype(float)
train_df['Cabin_side'] = train_df['Cabin'].str[-1:]
train_df['Cabin_side'] = train_df['Cabin_side'].astype(str)
test_df['Cabin_side'] = test_df['Cabin'].str[-1:]
test_df['Cabin_side'] = test_df['Cabin_side'].astype(str)
name_df = train_df['Name'].str.extract('(?P<First_Name>\\w+) (?P<Last_Name>\\w+)', expand=True)
train_df = pd.concat([train_df, name_df], axis=1)
train_df.head()
test_name_df = test_df['Name'].str.extract('(?P<First_Name>\\w+) (?P<Last_Name>\\w+)', expand=True)
test_df = pd.concat([test_df, test_name_df], axis=1)
test_df.head()

def destination(d):
    if d == 'TRAPPIST-1e':
        return 'Trappist'
    elif d == '55 Cancri e':
        return 'Cancri'
    else:
        return 'PSO'
train_df['Destination'] = train_df['Destination'].apply(destination)
test_df['Destination'] = test_df['Destination'].apply(destination)
print(train_df['Destination'].value_counts(dropna=False))
print(test_df['Destination'].value_counts(dropna=False))
train_df['Transported'] = train_df['Transported'].apply(int)
train_df['Transported']
train_df['CryoSleep'] = train_df['CryoSleep'].apply(str)
test_df['CryoSleep'] = test_df['CryoSleep'].apply(str)

def sleep(s):
    if s == 'False':
        return 'False'
    else:
        return 'True'
train_df['CryoSleep'] = train_df['CryoSleep'].apply(sleep)
test_df['CryoSleep'] = test_df['CryoSleep'].apply(sleep)
print(train_df['CryoSleep'].value_counts(dropna=False))
print(test_df['CryoSleep'].value_counts(dropna=False))
corr = train_df.corr()
corr
fig = px.imshow(corr, width=800, height=800, text_auto='.2f')
fig.update_xaxes(side='top')
fig.show()
not_transported_df = train_df[train_df['Transported'] == 0]
transported_df = train_df[train_df['Transported'] == 1]
transported_df.head()
not_transported_df.head()
fig = px.histogram(train_df, x='Transported', barmode='group', facet_col='HomePlanet')
fig.show()
fig = px.histogram(transported_df, x='CryoSleep', color='HomePlanet', barmode='group', title='Transported Passengers Cryo Sleep Status')
fig.show()
fig = px.histogram(transported_df, x='Cabin_side', color='HomePlanet', barmode='group', title='Transported Passengers Cabin info')
fig.show()
fig = px.histogram(transported_df, x='Cabin_deck', color='HomePlanet', barmode='group', title='Transported Passengers Cabin info')
fig.show()
fig = px.histogram(train_df, x='Destination', color='Transported', barmode='group', facet_col='HomePlanet')
fig.show()
fig = px.histogram(transported_df, x='Age', facet_col='HomePlanet', nbins=35)
fig.show()
fig = px.histogram(transported_df, x='VIP', facet_col='HomePlanet', barmode='group')
fig.show()
Transported = transported_df.groupby('PassengerId_p').count()['Transported']
Passengers = train_df.groupby('PassengerId_p').count()['Transported']
Percentage = Transported / Passengers * 100
Percentage
Cleaned_train_df = train_df.drop(columns=['PassengerId', 'Cabin', 'Name'])
Cleaned_test_df = test_df.drop(columns=['PassengerId', 'Cabin', 'Name'])
Cleaned_train_df.isnull().sum()
Cleaned_test_df.isnull().sum()
Cleaned_train_df['HomePlanet'] = Cleaned_train_df['HomePlanet'].fillna('Mars')
Cleaned_test_df['HomePlanet'] = Cleaned_test_df['HomePlanet'].fillna('Mars')

def planet(p):
    if p == 'Earth':
        return 0
    elif p == 'Europa':
        return 1
    else:
        return 2
Cleaned_train_df['HomePlanet'] = Cleaned_train_df['HomePlanet'].apply(planet)
Cleaned_test_df['HomePlanet'] = Cleaned_test_df['HomePlanet'].apply(planet)
Cleaned_train_df['HomePlanet'].value_counts(normalize=True, dropna=False)
Cleaned_test_df['HomePlanet'].value_counts(normalize=True, dropna=False)
Cleaned_train_df['VIP'] = Cleaned_train_df['VIP'].fillna(True)
Cleaned_test_df['VIP'] = Cleaned_test_df['VIP'].fillna(True)
Cleaned_train_df['VIP'] = Cleaned_train_df['VIP'].astype(int)
Cleaned_test_df['VIP'] = Cleaned_test_df['VIP'].astype(int)

def cabinside(c):
    if c == 'P':
        return 1
    else:
        return 0
Cleaned_train_df['Cabin_side'] = Cleaned_train_df['Cabin_side'].apply(cabinside)
Cleaned_test_df['Cabin_side'] = Cleaned_test_df['Cabin_side'].apply(cabinside)

def deck(d):
    if d == 'A':
        return 0
    elif d == 'B':
        return 1
    elif d == 'C':
        return 2
    elif d == 'D':
        return 3
    elif d == 'E':
        return 4
    elif d == 'F':
        return 5
    else:
        return 6
Cleaned_train_df['Cabin_deck'] = Cleaned_train_df['Cabin_deck'].apply(deck)
Cleaned_test_df['Cabin_deck'] = Cleaned_test_df['Cabin_deck'].apply(deck)
Cleaned_train_df.isnull().sum()
Cleaned_test_df.isnull().sum()
Cleaned_train_df.columns
Cleaned_train_df.info()
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn import metrics
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'PassengerId_g', 'PassengerId_p', 'Cabin_deck', 'Cabin_num', 'Cabin_side']
X = Cleaned_train_df[features]
y = Cleaned_train_df['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=22)
X_train.info()
eval_Xy = Pool(X_test, y_test, cat_features=[1, 2])
cat_model = CatBoostClassifier(iterations=4000, learning_rate=0.02, random_seed=22, max_depth=8, l2_leaf_reg=4, bootstrap_type='MVS', subsample=0.95, eval_metric='AUC', auto_class_weights='SqrtBalanced', early_stopping_rounds=1000)