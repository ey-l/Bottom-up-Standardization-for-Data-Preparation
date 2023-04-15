import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
data
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['PassengerId'] = le.fit_transform(data['PassengerId'])
data['PassengerId']
data.Transported.unique()
data.head(2).T
data.isnull().sum()
data.Destination = data.Destination.fillna('None')
data.CryoSleep = data.CryoSleep.fillna(True)
data.VIP = data.VIP.fillna(True)
data.Name = data.Name.fillna('anonymous')
data.Age = data.Age.fillna(20)
data.RoomService = data.RoomService.fillna(20)
data.FoodCourt = data.FoodCourt.fillna(0)
data.ShoppingMall = data.ShoppingMall.fillna(1000)
data.Spa = data.Spa.fillna(1000)
data.VRDeck = data.VRDeck.fillna(20)
data.HomePlanet = data.HomePlanet.fillna('None')
data['Destination'] = data['Destination'].replace('TRAPPIST-1e', 'TRAPPIST').replace('PSO J318.5-22', 'PSO').replace('55 Cancri e', 'Canceri')
data['CryoSleep'] = data['CryoSleep'].replace(True, 1).replace(False, 0)
data['VIP'] = data['VIP'].replace(True, 1).replace(False, 0)
data['Transported'] = data['Transported'].replace(False, 0).replace(True, 1)
data.head(10)
data.Cabin.unique
data = data.astype({'Age': 'int', 'RoomService': 'int', 'FoodCourt': 'int', 'ShoppingMall': 'int', 'VRDeck': 'int', 'Spa': 'int'})
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize': (15, 8)})
sns.countplot(x='HomePlanet', hue='Destination', data=data, palette='flare')
sns.countplot(data.Transported)
data.iloc[:, :-1].describe().T.sort_values(by='std', ascending=False).style.background_gradient(cmap='magma').bar(subset=['max'], color='#BB0000').bar(subset=['mean'], color='green')
x = data.drop(columns=['Cabin', 'Transported', 'Name'])
y = data.Transported
x.CryoSleep.unique()
x.isnull().sum()
x
data.HomePlanet.value_counts()
data.HomePlanet.unique()
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