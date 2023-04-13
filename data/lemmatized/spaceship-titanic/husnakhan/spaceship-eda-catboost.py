import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
_input1['PassengerId'] = le.fit_transform(_input1['PassengerId'])
_input1['PassengerId']
_input1.Transported.unique()
_input1.head(2).T
_input1.isnull().sum()
_input1.Destination = _input1.Destination.fillna('None')
_input1.CryoSleep = _input1.CryoSleep.fillna(True)
_input1.VIP = _input1.VIP.fillna(True)
_input1.Name = _input1.Name.fillna('anonymous')
_input1.Age = _input1.Age.fillna(20)
_input1.RoomService = _input1.RoomService.fillna(20)
_input1.FoodCourt = _input1.FoodCourt.fillna(0)
_input1.ShoppingMall = _input1.ShoppingMall.fillna(1000)
_input1.Spa = _input1.Spa.fillna(1000)
_input1.VRDeck = _input1.VRDeck.fillna(20)
_input1.HomePlanet = _input1.HomePlanet.fillna('None')
_input1['Destination'] = _input1['Destination'].replace('TRAPPIST-1e', 'TRAPPIST').replace('PSO J318.5-22', 'PSO').replace('55 Cancri e', 'Canceri')
_input1['CryoSleep'] = _input1['CryoSleep'].replace(True, 1).replace(False, 0)
_input1['VIP'] = _input1['VIP'].replace(True, 1).replace(False, 0)
_input1['Transported'] = _input1['Transported'].replace(False, 0).replace(True, 1)
_input1.head(10)
_input1.Cabin.unique
_input1 = _input1.astype({'Age': 'int', 'RoomService': 'int', 'FoodCourt': 'int', 'ShoppingMall': 'int', 'VRDeck': 'int', 'Spa': 'int'})
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize': (15, 8)})
sns.countplot(x='HomePlanet', hue='Destination', data=_input1, palette='flare')
sns.countplot(_input1.Transported)
_input1.iloc[:, :-1].describe().T.sort_values(by='std', ascending=False).style.background_gradient(cmap='magma').bar(subset=['max'], color='#BB0000').bar(subset=['mean'], color='green')
x = _input1.drop(columns=['Cabin', 'Transported', 'Name'])
y = _input1.Transported
x.CryoSleep.unique()
x.isnull().sum()
x
_input1.HomePlanet.value_counts()
_input1.HomePlanet.unique()
nom_cols = [1, 3]
num_cols = [4, 5, 6, 7, 8, 9]
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn import set_config
from sklearn.experimental import enable_iterative_imputer
from catboost import CatBoostClassifier
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import make_pipeline
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.35)
cat_features = list(range(0, x.shape[1]))
clf = CatBoostClassifier(iterations=350)