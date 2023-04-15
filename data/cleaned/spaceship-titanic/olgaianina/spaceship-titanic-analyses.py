import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_x = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_x = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_x.head()
train_x.info()
test_x.info()
print('Number of transported passengers (target=True) =', train_x['Transported'].sum())
print('Number of not transported passengers (target=False) =', train_x.shape[0] - train_x['Transported'].sum())
sns.histplot(x=train_x.HomePlanet, hue=train_x.Transported.astype(int), multiple='dodge', shrink=0.9)
sns.histplot(x=train_x.CryoSleep.astype(float), hue=train_x.Transported.astype(int), multiple='dodge')
sns.histplot(x=train_x.Destination, hue=train_x.Transported.astype(int), multiple='dodge', shrink=0.9)
plt.figure(figsize=(20, 10))
g = sns.histplot(x=train_x.Age, hue=train_x.Transported.astype(int), multiple='dodge', shrink=0.9, bins=40)
g.set_xticks(range(0, 80, 2))

sns.histplot(x=train_x.VIP.astype(float), hue=train_x.Transported.astype(int), multiple='dodge')
plt.figure(figsize=(20, 10))
sns.boxplot(data=train_x[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
train_x['NumberOfAmenities'] = train_x['RoomService'].astype(bool).astype(int) + train_x['FoodCourt'].astype(bool).astype(int) + train_x['ShoppingMall'].astype(bool).astype(int) + train_x['Spa'].astype(bool).astype(int) + train_x['VRDeck'].astype(bool).astype(int)
test_x['NumberOfAmenities'] = test_x['RoomService'].astype(bool).astype(int) + test_x['FoodCourt'].astype(bool).astype(int) + test_x['ShoppingMall'].astype(bool).astype(int) + test_x['Spa'].astype(bool).astype(int) + test_x['VRDeck'].astype(bool).astype(int)
train_x.head()
sns.histplot(x=train_x.NumberOfAmenities, hue=train_x.Transported.astype(int), multiple='dodge')
datasets = [train_x, test_x]
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
train_x.head()
train_x.CabinDeck.value_counts()
sns.histplot(x=train_x.CabinDeck, hue=train_x.Transported.astype(int), multiple='dodge', shrink=0.9)
train_x.CabinSide.value_counts()
sns.histplot(x=train_x.CabinSide, hue=train_x.Transported.astype(int), multiple='dodge', shrink=0.9)
train_x.CabinNumber.value_counts()
plt.figure(figsize=(20, 10))
g = sns.histplot(x=train_x.CabinNumber.astype(float), hue=train_x.Transported.astype(int), multiple='dodge', shrink=0.9, bins=40)

test_x_id = test_x['PassengerId']
train_y = train_x['Transported'].astype(int)
train_x = train_x.drop(['Transported'], axis=1)
train_x.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
test_x.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
train_x.info()
test_x.info()
train_x.describe()
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
num_cols = ['Age', 'CabinNumber']
num_cols_amount = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = [col for col in train_x.columns if col not in num_cols and col not in num_cols_amount]
num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
num_amount_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder())])
preproc = ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('num_amount', num_amount_transformer, num_cols_amount), ('cat', cat_transformer, cat_cols)])
train_x = pd.DataFrame(preproc.fit_transform(train_x))
train_x.columns = num_cols + num_cols_amount + cat_cols
test_x = pd.DataFrame(preproc.fit_transform(test_x))
test_x.columns = num_cols + num_cols_amount + cat_cols
train_x.head()
test_x.head()
plt.figure(figsize=(15, 15))
sns.heatmap(data=train_x.assign(train_y=train_y).corr(), annot=True)
sns.histplot(x=train_x.NumberOfAmenities, hue=train_x.CryoSleep, multiple='dodge', shrink=0.9)
train_x.drop(['CryoSleep'], axis=1, inplace=True)
test_x.drop(['CryoSleep'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
(train_x, val_x, train_y, val_y) = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
print(train_x.shape)
print(val_x.shape)
from sklearn.svm import SVC
svc = SVC(random_state=0, decision_function_shape='ovo', kernel='rbf', C=4.0, gamma=0.03)