import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.shape
_input1.Cabin.describe()
_input1.Cabin.value_counts()
_input1.Cabin.isnull().sum()
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
st.mode(_input1['CryoSleep'])[0][0]
_input1.CryoSleep = _input1.CryoSleep.fillna(st.mode(_input1['CryoSleep'])[0][0], inplace=False)
_input1.VIP = _input1.VIP.fillna(st.mode(_input1['VIP'])[0][0], inplace=False)
_input1['Transported'] = _input1['Transported'].astype(int)
_input1['CryoSleep'] = _input1['CryoSleep'].astype(int)
_input1['VIP'] = _input1['VIP'].astype(int)
_input1.head()
_input1.isnull().sum()
_input1.dtypes
_input1.HomePlanet = _input1.HomePlanet.replace(['Europa', 'Earth', 'Mars'], [0, 1, 2], inplace=False)
_input1.head()
_input1.Destination = _input1.Destination.replace(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], [0, 1, 2], inplace=False)
_input1.head()
_input1.isnull().sum()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
main_df = imputer.fit_transform(_input1)
main_df.shape
_input1 = pd.DataFrame(main_df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported'], dtype='int64')
_input1.head()
_input1.describe()
_input1.isnull().sum()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
X = _input1.iloc[:, :-1]
y = _input1.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1)