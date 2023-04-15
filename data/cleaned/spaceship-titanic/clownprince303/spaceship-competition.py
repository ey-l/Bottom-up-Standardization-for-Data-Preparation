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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.head(5)
train.isnull().sum()
imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
train.HomePlanet = imputer.fit_transform(train['HomePlanet'].values.reshape(-1, 1))[:, 0]
train.CryoSleep = imputer.fit_transform(train['CryoSleep'].values.reshape(-1, 1))[:, 0]
train.Destination = imputer.fit_transform(train['Destination'].values.reshape(-1, 1))[:, 0]
train.VIP = imputer.fit_transform(train['VIP'].values.reshape(-1, 1))[:, 0]
imputer2 = SimpleImputer(missing_values=np.NaN, strategy='mean')
train.Age = imputer2.fit_transform(train['Age'].values.reshape(-1, 1))[:, 0]
train.RoomService = imputer2.fit_transform(train['RoomService'].values.reshape(-1, 1))[:, 0]
train.FoodCourt = imputer2.fit_transform(train['FoodCourt'].values.reshape(-1, 1))[:, 0]
train.ShoppingMall = imputer2.fit_transform(train['ShoppingMall'].values.reshape(-1, 1))[:, 0]
train.Spa = imputer2.fit_transform(train['Spa'].values.reshape(-1, 1))[:, 0]
train.VRDeck = imputer2.fit_transform(train['VRDeck'].values.reshape(-1, 1))[:, 0]
train = train.drop('Name', axis=1)
train = train.drop('Cabin', axis=1)
train.isnull().sum()
train['Transported'].replace({False: 0, True: 1}, inplace=True)
train['CryoSleep'].replace({False: 0, True: 1}, inplace=True)
train['VIP'].replace({False: 0, True: 1}, inplace=True)
encoder = ce.OneHotEncoder(cols=['HomePlanet', 'Destination'], handle_unknown='return_nan', return_df=True, use_cat_names=True)
train = encoder.fit_transform(train)
train.head(5)
X = train.copy()
X = X.dropna()
Y = X.pop('Transported')
mi_scores = make_mi_scores(X, Y)
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
train['totalExp'] = train['ShoppingMall'] + train['Spa'] + train['FoodCourt'] + train['RoomService'] + train['VRDeck']
train.head(15)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
Y = train['Transported']
X = train[['Spa', 'CryoSleep', 'VRDeck', 'RoomService', 'FoodCourt', 'ShoppingMall']]
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, random_state=23)