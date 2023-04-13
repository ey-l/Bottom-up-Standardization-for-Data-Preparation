import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import category_encoders as ce
from sklearn import metrics

def make_mi_scores(X, y):
    X = X.copy()
    mi_scores = mutual_info_regression(X, y, discrete_features='auto', random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head(5)
_input1.isnull().sum()
imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
_input1.HomePlanet = imputer.fit_transform(_input1['HomePlanet'].values.reshape(-1, 1))[:, 0]
_input1.CryoSleep = imputer.fit_transform(_input1['CryoSleep'].values.reshape(-1, 1))[:, 0]
_input1.Destination = imputer.fit_transform(_input1['Destination'].values.reshape(-1, 1))[:, 0]
_input1.VIP = imputer.fit_transform(_input1['VIP'].values.reshape(-1, 1))[:, 0]
imputer2 = SimpleImputer(missing_values=np.NaN, strategy='mean')
_input1.Age = imputer2.fit_transform(_input1['Age'].values.reshape(-1, 1))[:, 0]
_input1.RoomService = imputer2.fit_transform(_input1['RoomService'].values.reshape(-1, 1))[:, 0]
_input1.FoodCourt = imputer2.fit_transform(_input1['FoodCourt'].values.reshape(-1, 1))[:, 0]
_input1.ShoppingMall = imputer2.fit_transform(_input1['ShoppingMall'].values.reshape(-1, 1))[:, 0]
_input1.Spa = imputer2.fit_transform(_input1['Spa'].values.reshape(-1, 1))[:, 0]
_input1.VRDeck = imputer2.fit_transform(_input1['VRDeck'].values.reshape(-1, 1))[:, 0]
_input1 = _input1.drop('Name', axis=1)
_input1 = _input1.drop('Cabin', axis=1)
_input1.isnull().sum()
_input1['Transported'] = _input1['Transported'].replace({False: 0, True: 1}, inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].replace({False: 0, True: 1}, inplace=False)
_input1['VIP'] = _input1['VIP'].replace({False: 0, True: 1}, inplace=False)
encoder = ce.OneHotEncoder(cols=['HomePlanet', 'Destination'], handle_unknown='return_nan', return_df=True, use_cat_names=True)
_input1 = encoder.fit_transform(_input1)
_input1.head(5)
X = _input1.copy()
X = X.dropna()
Y = X.pop('Transported')
mi_scores = make_mi_scores(X, Y)
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
_input1['totalExp'] = _input1['ShoppingMall'] + _input1['Spa'] + _input1['FoodCourt'] + _input1['RoomService'] + _input1['VRDeck']
_input1.head(15)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
Y = _input1['Transported']
X = _input1[['Spa', 'CryoSleep', 'VRDeck', 'RoomService', 'FoodCourt', 'ShoppingMall']]
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, random_state=23)