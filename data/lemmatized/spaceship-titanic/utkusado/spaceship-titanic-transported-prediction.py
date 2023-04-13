import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
_input1.info()
_input1.describe().T
plt.figure(figsize=(7, 7))
_input1['Transported'].value_counts().plot.pie(explode=[0.05, 0.05], autopct='%1.1f%%', textprops={'fontsize': 16})
plt.figure(figsize=(12, 7))
sns.histplot(data=_input1, x='Age', hue='Transported', binwidth=1, kde=True)
plt.xlabel('Age')
plt.ylabel('Transported Count')
graph_cat = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
fig = plt.figure(figsize=(15, 15))
for (i, name) in enumerate(graph_cat):
    ax = fig.add_subplot(4, 1, i + 1)
    sns.countplot(data=_input1, axes=ax, x=name, hue='Transported')
sns.distplot(_input1['RoomService'])
sns.distplot(np.log(_input1['RoomService'] + 1))
_input1.isnull().sum()
_input0.isnull().sum()
train_test = [_input1, _input0]
for data_age in train_test:
    mean = data_age['Age'].mean()
    std = data_age['Age'].std()
    is_null = data_age['Age'].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    age = data_age['Age'].copy()
    age[np.isnan(age)] = rand_age
    data_age['Age'] = age
    data_age['Age'] = data_age['Age'].astype(int)
for data_obj in train_test:
    data_obj['HomePlanet'] = data_obj['HomePlanet'].fillna(data_obj['HomePlanet'].mode()[0])
    data_obj['CryoSleep'] = data_obj['CryoSleep'].fillna(data_obj['CryoSleep'].mode()[0])
    data_obj['Cabin'] = data_obj['Cabin'].fillna(data_obj['Cabin'].mode()[0])
    data_obj['Destination'] = data_obj['Destination'].fillna(data_obj['Destination'].mode()[0])
    data_obj['VIP'] = data_obj['VIP'].fillna(data_obj['VIP'].mode()[0])
_input1.isnull().sum()
for data_num in train_test:
    data_num['RoomService'] = np.log(data_num['Spa'] + 1)
    data_num['ShoppingMall'] = np.log(data_num['ShoppingMall'] + 1)
    data_num['Spa'] = np.log(data_num['Spa'] + 1)
    data_num['VRDeck'] = np.log(data_num['VRDeck'] + 1)
    data_num['FoodCourt'] = np.log(data_num['FoodCourt'] + 1)
    data_num['RoomService'] = data_num['RoomService'].fillna(data_num['RoomService'].median())
    data_num['ShoppingMall'] = data_num['ShoppingMall'].fillna(data_num['ShoppingMall'].median())
    data_num['Spa'] = data_num['Spa'].fillna(data_num['Spa'].median())
    data_num['VRDeck'] = data_num['VRDeck'].fillna(data_num['VRDeck'].median())
    data_num['FoodCourt'] = data_num['FoodCourt'].fillna(data_num['FoodCourt'].median())
_input1.isnull().sum()
for data_name in train_test:
    data_name = data_name.drop('Name', axis=1, inplace=False)
_input1.isnull().sum()
for data in train_test:
    data['HomePlanet'] = data['HomePlanet'].astype('category').cat.codes
    data['Destination'] = data['Destination'].astype('category').cat.codes
corr = _input1.corr()
plt.figure(figsize=(15, 9))
sns.heatmap(corr, annot=True, cmap='coolwarm')
_input1 = _input1.drop('Cabin', axis=1)
_input0 = _input0.drop('Cabin', axis=1)
X = _input1.drop('Transported', axis=1)
y = _input1['Transported']
X_test = _input0
X.head()
y.head()
X_test.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)
from sklearn.model_selection import train_test_split, cross_val_score

def classify(model, xx, yy):
    (x_train, x_test, y_train, y_test) = train_test_split(xx, yy, test_size=0.2, random_state=42)