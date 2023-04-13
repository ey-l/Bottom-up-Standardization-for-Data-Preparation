import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.info()
_input0.info()
print('Number of transported passengers (target=True) =', _input1['Transported'].sum())
print('Number of not transported passengers (target=False) =', _input1.shape[0] - _input1['Transported'].sum())
sns.histplot(x=_input1.HomePlanet, hue=_input1.Transported.astype(int), multiple='dodge', shrink=0.9)
sns.histplot(x=_input1.CryoSleep.astype(float), hue=_input1.Transported.astype(int), multiple='dodge')
sns.histplot(x=_input1.Destination, hue=_input1.Transported.astype(int), multiple='dodge', shrink=0.9)
plt.figure(figsize=(20, 10))
g = sns.histplot(x=_input1.Age, hue=_input1.Transported.astype(int), multiple='dodge', shrink=0.9, bins=40)
g.set_xticks(range(0, 80, 2))
sns.histplot(x=_input1.VIP.astype(float), hue=_input1.Transported.astype(int), multiple='dodge')
plt.figure(figsize=(20, 10))
sns.boxplot(data=_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
_input1['NumberOfAmenities'] = _input1['RoomService'].astype(bool).astype(int) + _input1['FoodCourt'].astype(bool).astype(int) + _input1['ShoppingMall'].astype(bool).astype(int) + _input1['Spa'].astype(bool).astype(int) + _input1['VRDeck'].astype(bool).astype(int)
_input0['NumberOfAmenities'] = _input0['RoomService'].astype(bool).astype(int) + _input0['FoodCourt'].astype(bool).astype(int) + _input0['ShoppingMall'].astype(bool).astype(int) + _input0['Spa'].astype(bool).astype(int) + _input0['VRDeck'].astype(bool).astype(int)
_input1.head()
sns.histplot(x=_input1.NumberOfAmenities, hue=_input1.Transported.astype(int), multiple='dodge')
datasets = [_input1, _input0]
for d in datasets:
    nrows = d.shape[0]
    list_deck = [np.NaN for i in range(nrows)]
    list_side = [np.NaN for i in range(nrows)]
    list_number = [0 for i in range(nrows)]
    for i in range(nrows):
        if type(d['Cabin'][i]) == str:
            cab = d['Cabin'][i].split('/')
            list_deck[i] = cab[0]
            list_number[i] = cab[1]
            list_side[i] = cab[2]
    d['CabinDeck'] = list_deck
    d['CabinSide'] = list_side
    d['CabinNumber'] = list_number
    list_ppl = [0 for i in range(nrows)]
    next_i = 0
    for i in range(nrows):
        if i < next_i and num_ppl != 0:
            continue
        a = d.iloc[i, 0].split('_')[0]
        next_i = i + 1
        num_ppl = 0
        while True:
            if next_i >= nrows:
                break
            b = d.iloc[next_i, 0].split('_')[0]
            if a != b:
                break
            num_ppl += 1
            next_i += 1
        if num_ppl > 0:
            for ind in range(i, next_i):
                list_ppl[ind] = num_ppl
    d['IsAlone'] = [int(not bool(l)) for l in list_ppl]
_input1.head()
_input1.CabinDeck.value_counts()
sns.histplot(x=_input1.CabinDeck, hue=_input1.Transported.astype(int), multiple='dodge', shrink=0.9)
_input1.CabinSide.value_counts()
sns.histplot(x=_input1.CabinSide, hue=_input1.Transported.astype(int), multiple='dodge', shrink=0.9)
_input1.CabinNumber.value_counts()
plt.figure(figsize=(20, 10))
g = sns.histplot(x=_input1.CabinNumber.astype(float), hue=_input1.Transported.astype(int), multiple='dodge', shrink=0.9, bins=40)
test_x_id = _input0['PassengerId']
train_y = _input1['Transported'].astype(int)
_input1 = _input1.drop(['Transported'], axis=1)
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
_input0 = _input0.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
_input1.info()
_input0.info()
_input1.describe()
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
num_cols = ['Age', 'CabinNumber']
num_cols_amount = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = [col for col in _input1.columns if col not in num_cols and col not in num_cols_amount]
num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
num_amount_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder())])
preproc = ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('num_amount', num_amount_transformer, num_cols_amount), ('cat', cat_transformer, cat_cols)])
_input1 = pd.DataFrame(preproc.fit_transform(_input1))
_input1.columns = num_cols + num_cols_amount + cat_cols
_input0 = pd.DataFrame(preproc.fit_transform(_input0))
_input0.columns = num_cols + num_cols_amount + cat_cols
_input1.head()
_input0.head()
plt.figure(figsize=(15, 15))
sns.heatmap(data=_input1.assign(train_y=train_y).corr(), annot=True)
sns.histplot(x=_input1.NumberOfAmenities, hue=_input1.CryoSleep, multiple='dodge', shrink=0.9)
_input1 = _input1.drop(['CryoSleep'], axis=1, inplace=False)
_input0 = _input0.drop(['CryoSleep'], axis=1, inplace=False)
from sklearn.model_selection import train_test_split
(_input1, val_x, train_y, val_y) = train_test_split(_input1, train_y, test_size=0.2, random_state=0)
print(_input1.shape)
print(val_x.shape)
from sklearn.svm import SVC
svc = SVC(random_state=0, decision_function_shape='ovo', kernel='rbf', C=4.0, gamma=0.03)