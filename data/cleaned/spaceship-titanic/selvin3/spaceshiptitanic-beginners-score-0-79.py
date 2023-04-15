import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head()
df.shape
df.Cabin.describe()
df.Cabin.value_counts()
df.Cabin.isnull().sum()
df.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
st.mode(df['CryoSleep'])[0][0]
df.CryoSleep.fillna(st.mode(df['CryoSleep'])[0][0], inplace=True)
df.VIP.fillna(st.mode(df['VIP'])[0][0], inplace=True)
df['Transported'] = df['Transported'].astype(int)
df['CryoSleep'] = df['CryoSleep'].astype(int)
df['VIP'] = df['VIP'].astype(int)
df.head()
df.isnull().sum()
df.dtypes
df.HomePlanet.replace(['Europa', 'Earth', 'Mars'], [0, 1, 2], inplace=True)
df.head()
df.Destination.replace(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], [0, 1, 2], inplace=True)
df.head()
df.isnull().sum()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
main_df = imputer.fit_transform(df)
main_df.shape
df = pd.DataFrame(main_df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported'], dtype='int64')
df.head()
df.describe()
df.isnull().sum()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1)