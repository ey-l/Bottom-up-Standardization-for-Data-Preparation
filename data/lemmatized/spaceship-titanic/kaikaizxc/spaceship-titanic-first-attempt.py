import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.sample(5)
_input1.info()
_input1.describe()
for row in _input1.columns:
    print('Percentage of {} missing is: {}'.format(row, _input1[row].isnull().sum() / len(_input1) * 100))
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.isnull().sum()
dataframes = [_input1, _input0]
for data in dataframes:
    data['HomePlanet'] = data['HomePlanet'].fillna(data['HomePlanet'].mode()[0], inplace=False)
    data['CryoSleep'] = data['CryoSleep'].fillna(data['CryoSleep'].mode()[0], inplace=False)
    data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0], inplace=False)
    data['Destination'] = data['Destination'].fillna(data['Destination'].mode()[0], inplace=False)
    data['Age'] = data['Age'].fillna(data['Age'].median(), inplace=False)
    data['VIP'] = data['VIP'].fillna(data['VIP'].mode()[0], inplace=False)
    data['RoomService'] = data['RoomService'].fillna(data['RoomService'].mean(), inplace=False)
    data['FoodCourt'] = data['FoodCourt'].fillna(data['FoodCourt'].mean(), inplace=False)
    data['ShoppingMall'] = data['ShoppingMall'].fillna(data['ShoppingMall'].mean(), inplace=False)
    data['Spa'] = data['Spa'].fillna(data['Spa'].mean(), inplace=False)
    data['VRDeck'] = data['VRDeck'].fillna(data['VRDeck'].mean(), inplace=False)
    data['Name'] = data['Name'].fillna(data['Name'].mode()[0], inplace=False)
_input1.isnull().sum()

def getGroup(PassengerId):
    return PassengerId.split('_')[0]

def getDeck(Cabin):
    return Cabin.split('/')[0]

def getPosition(Cabin):
    return Cabin.split('/')[-1]
for data in dataframes:
    data['Age_Categories'] = pd.cut(data['Age'], bins=[0, 10, 20, 60, np.inf], labels=['Child', 'Teenager', 'Adults', 'Elderly'])
    data['Passenger_Group'] = data['PassengerId'].apply(lambda x: getGroup(x))
    data['Deck'] = data['Cabin'].apply(lambda x: getDeck(x))
    data['Position'] = data['Cabin'].apply(lambda x: getPosition(x))
    data['CryoSleep'] = data['CryoSleep'].apply(lambda x: 0 if x == False else 1)
    data['VIP'] = data['VIP'].apply(lambda x: 0 if x == False else 1)
_input1.sample(10)
print('Passenger Groups')
print(len(_input1['Passenger_Group'].unique()))
print('=' * 20)
print('Destination')
print(len(_input1['Destination'].unique()))
print('=' * 20)
print('Deck')
print(len(_input1['Deck'].unique()))
print('=' * 20)
_input1['Transported'] = _input1['Transported'].apply(lambda x: 0 if x == False else 1)
plt.figure(figsize=(12, 12))
sns.pairplot(_input1)
sns.countplot(x='Age_Categories', hue='Transported', data=_input1)
plt.title('Count of different Age Categories with transported')
sns.boxplot(x='Age', data=_input1)
sns.displot(x='Age', data=_input1, kde=True)
plt.title('Distribution of ages')
sns.countplot(x='HomePlanet', hue='Transported', data=_input1)
sns.countplot(x='CryoSleep', hue='Transported', data=_input1)
sns.countplot(x='Destination', hue='Transported', data=_input1)
sns.countplot(x='VIP', hue='Transported', data=_input1)
sns.boxplot(x=_input1['RoomService'])
sns.violinplot(x=_input1['RoomService'])
sns.boxplot(x=_input1['FoodCourt'])
sns.violinplot(x=_input1['FoodCourt'])
sns.boxplot(x=_input1['ShoppingMall'])
sns.violinplot(x=_input1['ShoppingMall'])
sns.boxplot(x=_input1['Spa'])
sns.violinplot(x=_input1['Spa'])
sns.boxplot(x=_input1['VRDeck'])
sns.violinplot(x=_input1['VRDeck'])

def getValidRange(feature, data):
    (q1, q3) = np.percentile(data[feature], [25, 75])
    iqr = q3 - q1
    return [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
features_to_edit = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for feature in features_to_edit:
    (lower_fence, upper_fence) = getValidRange(feature, _input1)
    print(feature, len(data[feature].loc[(data[feature] > upper_fence) | (data[feature] < lower_fence)]))
print('total', len(_input1))
sns.countplot(x='Deck', hue='Transported', data=_input1)
sns.countplot(x='Position', hue='Transported', data=_input1)
y = _input1['Transported']
x = _input1[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'Deck', 'Position', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Position']
num_pipeline = Pipeline([('Min_max', StandardScaler())])
cat_pipeline = Pipeline([('one_hot', OneHotEncoder())])
pipeline = ColumnTransformer([('num', num_pipeline, num_features), ('cat', cat_pipeline, cat_features)])
x = pipeline.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
lin_reg = LogisticRegression()