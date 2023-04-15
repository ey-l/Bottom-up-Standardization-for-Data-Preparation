import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_data.head()
train_data.shape
train_data.isna().sum()
train_data.describe().T
categorical_variables = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
mode = train_data[categorical_variables].mode().iloc[0]
train_data[categorical_variables] = train_data[categorical_variables].fillna(mode)
train_data.isna().sum()
continous_variables = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
median = train_data[continous_variables].median()
median
train_data[continous_variables] = train_data[continous_variables].fillna(median)
train_data.isna().sum()
train_data = train_data.drop('Name', axis=1)
train_data
train_data[['Deck', 'Num', 'Side']] = train_data['Cabin'].str.split('/', expand=True)
train_data
train_data = train_data.drop('Cabin', axis=1)
train_data
train_data.hist('Age')
train_data.Age.describe()
labels = ['Child', 'Teenager', 'Adult', 'Older']
bins = [0, 12, 21, 45, 80]
train_data['Age_Group'] = pd.cut(train_data['Age'], bins=bins, labels=labels)
train_data.head()
train_data = train_data.drop('Age', axis=1)
train_data.head()
lbe = LabelEncoder()
categorical_vars = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Num', 'Side', 'Age_Group', 'Transported']
train_data[categorical_vars] = train_data[categorical_vars].apply(lbe.fit_transform)
train_data
train_data.describe().T
sns.boxplot(data=train_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
tenth_percentile = np.percentile(train_data['RoomService'], 10)
ninetyseventh_percentile = np.percentile(train_data['RoomService'], 97)
print(f'10% - {tenth_percentile}\n97% - {ninetyseventh_percentile}')
train_data[train_data.RoomService > 1800.0].shape
y = train_data.Transported
X = train_data.drop(['Transported', 'PassengerId'], axis=1)
y.value_counts()
(X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y, random_state=127)
print(f'Shape of X_Train: {X_train.shape}\nShape of y_Train: {y_train.shape}\nShape of X_Test: {X_test.shape}\nShape of y_test: {y_test.shape}')
dt_classifier = DecisionTreeClassifier()