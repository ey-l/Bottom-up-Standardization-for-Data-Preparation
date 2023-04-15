import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import warnings
from termcolor import colored
train0 = pd.read_csv('data/input/spaceship-titanic/train.csv')
test0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
train0.head()
print(colored('Train Data', attrs=['bold']))
print('-------' * 6)
train0.info()
print('-------' * 6)
test0.head()
print(colored('Test Data', attrs=['bold']))
print('-------' * 6)
test0.info()
print('-------' * 6)
train = train0.drop(['Name', 'PassengerId', 'Cabin'], axis=1)
test = test0.drop(['Name', 'PassengerId', 'Cabin'], axis=1)
train.head()
test.head()
plt.figure(figsize=(10, 6))
sns.countplot(x=train['Transported'], palette='Blues_r')
nan_cols = train.columns.tolist()
plt.figure(figsize=(16, 8))
plt.title('Missing Values in the Train Data')
nan_count_cols = train[nan_cols].isna().sum()
print('Missing values in train dataset:')
print('------' * 6)
print(nan_count_cols)
print('------' * 6)
sns.barplot(y=nan_count_cols, x=nan_cols, palette='Blues_r')

nan_cols = test.columns.tolist()
plt.figure(figsize=(16, 8))
plt.title('Missing Values in the Test Data')
nan_count_cols = test[nan_cols].isna().sum()
print('Missing values in test dataset:')
print('------' * 6)
print(nan_count_cols)
print('------' * 6)
sns.barplot(y=nan_count_cols, x=nan_cols, palette='Blues_r')

LABELS = train.columns
for col in LABELS:
    if col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        train[col].fillna(train[col].median(), inplace=True)
    else:
        train[col].fillna(train[col].mode()[0], inplace=True)
train.head()
nan_cols = train.columns.tolist()
plt.figure(figsize=(16, 8))
nan_count_cols = train[nan_cols].isna().sum()
print('Missing values in train dataset:')
print('------' * 6)
print(nan_count_cols)
print('------' * 6)
LABELS = test.columns
for col in LABELS:
    if col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        test[col].fillna(test[col].median(), inplace=True)
    else:
        test[col].fillna(test[col].mode()[0], inplace=True)
train.head()
nan_cols = test.columns.tolist()
plt.figure(figsize=(16, 8))
nan_count_cols = test[nan_cols].isna().sum()
print('Missing values in test dataset:')
print('------' * 6)
print(nan_count_cols)
print('------' * 6)
le = LabelEncoder()
train.HomePlanet = le.fit_transform(train.HomePlanet)
train.CryoSleep = le.fit_transform(train.CryoSleep)
train.Destination = le.fit_transform(train.Destination)
train.VIP = le.fit_transform(train.VIP)
train.Transported = le.fit_transform(train.Transported)
train.head()
le = LabelEncoder()
test.HomePlanet = le.fit_transform(test.HomePlanet)
test.CryoSleep = le.fit_transform(test.CryoSleep)
test.Destination = le.fit_transform(test.Destination)
test.VIP = le.fit_transform(test.VIP)
test.head()
plt.figure(figsize=(16, 8))
sns.heatmap(train.corr(), annot=True, cmap='Blues')
var = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for value in var:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(train[value], shade=True, bw=0.05)
    warnings.filterwarnings('ignore')

train_data = train.copy()
columns1 = ['FoodCourt', 'Spa', 'VRDeck', 'ShoppingMall', 'RoomService']
columns2 = ['Age']
for column in columns1:
    train_data[column] = StandardScaler().fit_transform(np.array(train_data[column]).reshape(-1, 1))
for column in columns2:
    train_data[column] = MinMaxScaler().fit_transform(np.array(train_data[column]).reshape(-1, 1))
train_data = train_data
train_data.head()
test_nor = test.copy()
columns = ['FoodCourt', 'Spa', 'VRDeck', 'ShoppingMall', 'RoomService']
columns2 = ['Age']
for column in columns:
    test_nor[column] = StandardScaler().fit_transform(np.array(test_nor[column]).reshape(-1, 1))
for column in columns2:
    train_data[column] = MinMaxScaler().fit_transform(np.array(train_data[column]).reshape(-1, 1))
test_nor = test_nor
test_nor.head()
X = train_data.drop(['Transported'], axis=1)
y = train_data['Transported']
(x_train, x_val, y_train, y_val) = train_test_split(X, y, test_size=0.05, random_state=42)
classifier2 = LogisticRegression(random_state=42)