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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
_input1.info()
_input0.info()
_input1.shape
_input0.shape
_input1.describe()
_input0.describe()
_input1.isna().sum()
_input0.isna().sum()
_input1.nunique()
_input0.nunique()
billed = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1['Total_billed'] = _input1[billed].sum(axis=1)
_input0['Total_billed'] = _input0[billed].sum(axis=1)
_input1.Total_billed.describe()
_input0.Total_billed.describe()
for i in billed:
    _input1[i] = _input1[i].fillna(_input1[i].median())
    _input0[i] = _input0[i].fillna(_input1[i].median())
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median())
_input0['Age'] = _input0['Age'].fillna(_input1['Age'].median())
_input1.info()
cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
for i in cols:
    _input1[i] = _input1[i].fillna(st.mode(_input1[i]))
    _input0[i] = _input0[i].fillna(st.mode(_input0[i]))
_input1.isna().sum()
X = _input1.drop(['Transported', 'Cabin', 'Total_billed', 'Name'], axis=1)
y = _input1.Transported
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1)
test = _input0.drop(labels=['Cabin', 'Name', 'Total_billed'], axis=1)
X_train.shape
num_col = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'PassengerId']
_input1[num_col] = _input1[num_col].astype(dtype='int')
_input0[num_col] = _input0[num_col].astype(dtype='int')
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