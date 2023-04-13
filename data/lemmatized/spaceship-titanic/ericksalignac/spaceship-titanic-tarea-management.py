import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import h2o
from h2o.automl import H2OAutoML
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', dtype={'HomePlanet': 'category', 'Cabin': 'string', 'Destination': 'category', 'Age': 'float32', 'RoomService': 'float32', 'FoodCourt': 'float32', 'ShoppingMall': 'float32', 'Spa': 'float32', 'VRDeck': 'float32', 'Transported': 'int'})
_input1 = _input1.drop(['Name'], axis=1, inplace=False)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', dtype={'HomePlanet': 'category', 'Cabin': 'string', 'Destination': 'category', 'Age': 'float32', 'RoomService': 'float32', 'FoodCourt': 'float32', 'ShoppingMall': 'float32', 'Spa': 'float32', 'VRDeck': 'float32', 'Transported': 'int'})
_input0 = _input0.drop(['Name'], axis=1, inplace=False)
_input1 = _input1.append(_input0)
_input1 = _input1.set_index('PassengerId', inplace=False)
_input1.head()
_input1.info()
_input1['Total'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input1
_input1.isna().sum()
nanrows = _input1.isna().any(axis=1).sum()
totalrows = _input1.shape[0]
nanrows / totalrows
_input1['HomePlanet'].isnull().sum()
sns.countplot(x=_input1['HomePlanet'])
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth', inplace=False)
_input1['HomePlanet'].isnull().sum()
sns.countplot(x=_input1['CryoSleep'])
_input1['CryoSleep'].isnull().sum()
nancryosleepers = (_input1['CryoSleep'].isnull() == True) & (_input1['Total'] == 0)
_input1.loc[nancryosleepers, 'CryoSleep'] = _input1.loc[nancryosleepers, 'CryoSleep'].fillna(True)
_input1['CryoSleep'].isnull().sum()
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False, inplace=False)
_input1['CryoSleep'].isnull().sum()
_input1[(_input1['CryoSleep'].isnull() == True) & (_input1['Total'] == 0)]
_input1['CryoSleep'].isna().value_counts()
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False, inplace=False)
_input1['CryoSleep'].isnull().sum()
_input1[['Deck', 'Number', 'Side']] = _input1['Cabin'].str.split('/', expand=True)
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input1
sns.countplot(x=_input1['Side'])
_input1['Side'] = _input1['Side'].fillna('S', inplace=False)
_input1['Side'].isnull().sum()
_input1['Deck'].isnull().sum()
sns.countplot(x=_input1['Deck'])
_input1['Deck'] = _input1['Deck'].fillna('F', inplace=False)
_input1['Deck'].isnull().sum()
_input1[['Deck', 'Side']] = _input1[['Deck', 'Side']].astype('category')
_input1.info()
_input1['Number'].isnull().sum()
_input1['Number'] = _input1['Number'].fillna('0', inplace=False)
_input1['Number'] = _input1['Number'].astype('int')
_input1.info()
plt.hist(x=_input1['Number'])
sns.countplot(x=_input1['Destination'])
_input1['Destination'].isnull().sum()
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e', inplace=False)
_input1['Destination'].isnull().sum()
plt.hist(x=_input1['Age'])
_input1['Age'].isnull().sum()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1['Age'].isnull().sum()
sns.countplot(x=_input1['VIP'])
_input1['VIP'].isnull().sum()
_input1['VIP'] = _input1['VIP'].fillna(False, inplace=False)
_input1['VIP'].isnull().sum()
grafico = px.histogram(_input1[_input1['RoomService'] > 0], x='RoomService')
grafico.show()
_input1['RoomService'].isnull().sum()
grafico = px.histogram(_input1[_input1['FoodCourt'] > 0], x='FoodCourt')
grafico.show()
_input1['FoodCourt'].isnull().sum()
grafico = px.histogram(_input1[_input1['ShoppingMall'] > 0], x='ShoppingMall')
grafico.show()
_input1['ShoppingMall'].isnull().sum()
grafico = px.histogram(_input1[_input1['Spa'] > 0], x='Spa')
grafico.show()
_input1['ShoppingMall'].isnull().sum()
grafico = px.histogram(_input1[_input1['VRDeck'] > 0], x='VRDeck')
grafico.show()
_input1['ShoppingMall'].isnull().sum()
spendings = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
iscryosleep = _input1['CryoSleep'] == True
awake = _input1['CryoSleep'] == False
for column in _input1[spendings]:
    _input1.loc[iscryosleep, column] = _input1.loc[iscryosleep, column].fillna(0)
    _input1.loc[awake, column] = _input1.loc[awake, column].fillna(_input1[column].mean())
_input1['RoomService'].isnull().sum()
_input1['FoodCourt'].isnull().sum()
_input1['ShoppingMall'].isnull().sum()
_input1['Spa'].isnull().sum()
_input1['VRDeck'].isnull().sum()
grafic = px.treemap(_input1, path=['CryoSleep', 'RoomService'])
grafic.show()
grafic = px.treemap(_input1, path=['CryoSleep', 'FoodCourt'])
grafic.show()
grafic = px.treemap(_input1, path=['CryoSleep', 'ShoppingMall'])
grafic.show()
grafic = px.treemap(_input1, path=['CryoSleep', 'Spa'])
grafic.show()
grafic = px.treemap(_input1, path=['CryoSleep', 'VRDeck'])
grafic.show()
_input1
X = _input1.drop('Transported', axis=1)
X
Y = _input1['Transported']
Y
label_encoder_cryo_sleep = LabelEncoder()
label_encoder_vip = LabelEncoder()
label_encoder_side = LabelEncoder()
X.iloc[:, 1] = label_encoder_cryo_sleep.fit_transform(X.iloc[:, 1])
X.iloc[:, 4] = label_encoder_vip.fit_transform(X.iloc[:, 4])
X.iloc[:, -1] = label_encoder_side.fit_transform(X.iloc[:, -1])
X
onehotencoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 2, 11])], remainder='passthrough')
X = pd.DataFrame(onehotencoder.fit_transform(X), columns=onehotencoder.get_feature_names_out(), index=_input1.index)
X.columns = [column.split('__')[1] for column in X.columns]
X
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=_input1.index)
X
Y.isna().sum()
_input0.shape
X_train = X[:-4277]
X_train
X_test = X[-4277:]
X_test
Y_train = Y[:-4277]
Y_train
Y_test = Y[-4277:]
Y_test
X_train = X_train.drop('Total', axis=1, inplace=False)
X_test = X_test.drop('Total', axis=1, inplace=False)

def new_model(model, **kwargs):
    new_model = model(**kwargs)