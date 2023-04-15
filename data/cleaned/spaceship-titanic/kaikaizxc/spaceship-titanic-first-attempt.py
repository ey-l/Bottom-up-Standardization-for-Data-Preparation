import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train.sample(5)
df_train.info()
df_train.describe()
for row in df_train.columns:
    print('Percentage of {} missing is: {}'.format(row, df_train[row].isnull().sum() / len(df_train) * 100))
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test.isnull().sum()
dataframes = [df_train, df_test]
for data in dataframes:
    data['HomePlanet'].fillna(data['HomePlanet'].mode()[0], inplace=True)
    data['CryoSleep'].fillna(data['CryoSleep'].mode()[0], inplace=True)
    data['Cabin'].fillna(data['Cabin'].mode()[0], inplace=True)
    data['Destination'].fillna(data['Destination'].mode()[0], inplace=True)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['VIP'].fillna(data['VIP'].mode()[0], inplace=True)
    data['RoomService'].fillna(data['RoomService'].mean(), inplace=True)
    data['FoodCourt'].fillna(data['FoodCourt'].mean(), inplace=True)
    data['ShoppingMall'].fillna(data['ShoppingMall'].mean(), inplace=True)
    data['Spa'].fillna(data['Spa'].mean(), inplace=True)
    data['VRDeck'].fillna(data['VRDeck'].mean(), inplace=True)
    data['Name'].fillna(data['Name'].mode()[0], inplace=True)
df_train.isnull().sum()

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
df_train.sample(10)
print('Passenger Groups')
print(len(df_train['Passenger_Group'].unique()))
print('=' * 20)
print('Destination')
print(len(df_train['Destination'].unique()))
print('=' * 20)
print('Deck')
print(len(df_train['Deck'].unique()))
print('=' * 20)
df_train['Transported'] = df_train['Transported'].apply(lambda x: 0 if x == False else 1)
plt.figure(figsize=(12, 12))
sns.pairplot(df_train)

sns.countplot(x='Age_Categories', hue='Transported', data=df_train)
plt.title('Count of different Age Categories with transported')

sns.boxplot(x='Age', data=df_train)

sns.displot(x='Age', data=df_train, kde=True)
plt.title('Distribution of ages')

sns.countplot(x='HomePlanet', hue='Transported', data=df_train)

sns.countplot(x='CryoSleep', hue='Transported', data=df_train)

sns.countplot(x='Destination', hue='Transported', data=df_train)

sns.countplot(x='VIP', hue='Transported', data=df_train)

sns.boxplot(x=df_train['RoomService'])

sns.violinplot(x=df_train['RoomService'])

sns.boxplot(x=df_train['FoodCourt'])

sns.violinplot(x=df_train['FoodCourt'])

sns.boxplot(x=df_train['ShoppingMall'])

sns.violinplot(x=df_train['ShoppingMall'])

sns.boxplot(x=df_train['Spa'])

sns.violinplot(x=df_train['Spa'])

sns.boxplot(x=df_train['VRDeck'])

sns.violinplot(x=df_train['VRDeck'])


def getValidRange(feature, data):
    (q1, q3) = np.percentile(data[feature], [25, 75])
    iqr = q3 - q1
    return [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
features_to_edit = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for feature in features_to_edit:
    (lower_fence, upper_fence) = getValidRange(feature, df_train)
    print(feature, len(data[feature].loc[(data[feature] > upper_fence) | (data[feature] < lower_fence)]))
print('total', len(df_train))
sns.countplot(x='Deck', hue='Transported', data=df_train)

sns.countplot(x='Position', hue='Transported', data=df_train)

y = df_train['Transported']
x = df_train[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'Deck', 'Position', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
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