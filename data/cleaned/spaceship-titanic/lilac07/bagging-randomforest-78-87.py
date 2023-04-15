import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
data_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
data_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
data_test
data_train.info()
data_train.isnull().sum()
data_train['HomePlanet'] = data_train['HomePlanet'].fillna('Earth')
data_test['HomePlanet'] = data_test['HomePlanet'].fillna('Earth')
data_train.isnull().sum().sort_values(ascending=False)
data_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = data_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
data_test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = data_test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())
data_train['VIP'].value_counts()
data_train['Destination'] = data_train['Destination'].fillna('PSO J318.5-22')
data_test['Destination'] = data_test['Destination'].fillna('PSO J318.5-22')
data_train['CryoSleep'] = data_train['CryoSleep'].fillna(False)
data_test['CryoSleep'] = data_test['CryoSleep'].fillna(False)
data_train['Cabin'] = data_train['Cabin'].fillna('T/0/P')
data_test['Cabin'] = data_test['Cabin'].fillna('T/0/P')
data_train.isnull().sum().sort_values(ascending=False)
data_test = data_test.drop(['Name'], axis=1)
data_train['VIP'] = data_train['VIP'].fillna(False)
data_test['VIP'] = data_test['VIP'].fillna(False)
data_train.corr()['Transported'].sort_values(ascending=False)[1:]
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
train_x = data_train.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
imputer = KNNImputer(n_neighbors=5)
train_x = imputer.fit_transform(train_x)
train_x = pd.DataFrame(train_x)
print(train_x)
train_x.columns = data_train.columns.values
train_x.head()
type(train_x)
train_x.isnull().sum()
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
x_test = data_test.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
imputer = KNNImputer(n_neighbors=3)
x_test = imputer.fit_transform(x_test)
x_test = pd.DataFrame(x_test)
print(x_test)
x_test.columns = data_test.columns.values
x_test.head()
x_test.shape
from sklearn import model_selection
X = train_x.drop('Transported', axis=1)
y = train_x['Transported']
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X.drop(['Age', 'Name', 'PassengerId'], axis=1), y, test_size=0.1, random_state=123)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
dec_tree_cls = RandomForestClassifier(max_depth=8, bootstrap=True, n_estimators=250, oob_score=True)
bag_cls = BaggingClassifier(base_estimator=dec_tree_cls, random_state=42)