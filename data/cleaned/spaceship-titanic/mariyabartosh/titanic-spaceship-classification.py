import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import make_column_selector
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head()
df_test.head()
df_train.info()
df_test.info()
df_train.shape
df_test.shape
df_train.describe()
df_test.describe()
df_train.isna().sum()
df_test.isna().sum()
df_train.nunique()
df_test.nunique()
billed = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df_train['Total_billed'] = df_train[billed].sum(axis=1)
df_test['Total_billed'] = df_test[billed].sum(axis=1)
df_train.Total_billed.describe()
df_test.Total_billed.describe()
for i in billed:
    df_train[i] = df_train[i].fillna(df_train[i].median())
    df_test[i] = df_test[i].fillna(df_train[i].median())
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_test['Age'] = df_test['Age'].fillna(df_train['Age'].median())
df_train.info()
cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
for i in cols:
    df_train[i] = df_train[i].fillna(st.mode(df_train[i]))
    df_test[i] = df_test[i].fillna(st.mode(df_test[i]))
df_train.isna().sum()
X = df_train.drop(['Transported', 'Cabin', 'Total_billed', 'Name'], axis=1)
y = df_train.Transported
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1)
test = df_test.drop(labels=['Cabin', 'Name', 'Total_billed'], axis=1)
X_train.shape
num_col = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'PassengerId']
df_train[num_col] = df_train[num_col].astype(dtype='int')
df_test[num_col] = df_test[num_col].astype(dtype='int')
le = LabelEncoder()
for i in cat_cols:
    X_train[i] = le.fit_transform(X_train[i])
    X_test[i] = le.fit_transform(X_test[i])
    test[i] = le.fit_transform(test[i])
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
X_train_LR = X_train
X_test_LR = X_test
test_LR = test
mm = MinMaxScaler()
X_train_LR[num_col] = mm.fit_transform(X_train[num_col])
X_test_LR[num_col] = mm.transform(X_test[num_col])
test_LR[num_col] = mm.transform(test[num_col])
sc = StandardScaler()
X_train[num_col] = sc.fit_transform(X_train[num_col])
X_test[num_col] = sc.transform(X_test[num_col])
test[num_col] = sc.transform(test[num_col])
model_LR = LogisticRegression()