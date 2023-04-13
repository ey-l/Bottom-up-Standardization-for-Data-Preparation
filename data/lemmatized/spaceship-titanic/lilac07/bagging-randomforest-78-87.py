import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
_input1.info()
_input1.isnull().sum()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth')
_input1.isnull().sum().sort_values(ascending=False)
_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
_input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median())
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median())
_input1['VIP'].value_counts()
_input1['Destination'] = _input1['Destination'].fillna('PSO J318.5-22')
_input0['Destination'] = _input0['Destination'].fillna('PSO J318.5-22')
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(False)
_input1['Cabin'] = _input1['Cabin'].fillna('T/0/P')
_input0['Cabin'] = _input0['Cabin'].fillna('T/0/P')
_input1.isnull().sum().sort_values(ascending=False)
_input0 = _input0.drop(['Name'], axis=1)
_input1['VIP'] = _input1['VIP'].fillna(False)
_input0['VIP'] = _input0['VIP'].fillna(False)
_input1.corr()['Transported'].sort_values(ascending=False)[1:]
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
train_x = _input1.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
imputer = KNNImputer(n_neighbors=5)
train_x = imputer.fit_transform(train_x)
train_x = pd.DataFrame(train_x)
print(train_x)
train_x.columns = _input1.columns.values
train_x.head()
type(train_x)
train_x.isnull().sum()
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
x_test = _input0.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
imputer = KNNImputer(n_neighbors=3)
x_test = imputer.fit_transform(x_test)
x_test = pd.DataFrame(x_test)
print(x_test)
x_test.columns = _input0.columns.values
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