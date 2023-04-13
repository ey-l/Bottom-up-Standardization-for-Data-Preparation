import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
import seaborn as sns
sns.heatmap(_input1.isnull())
from sklearn import preprocessing
Lb = preprocessing.LabelEncoder()
_input1['PassengerId'] = Lb.fit_transform(_input1['PassengerId'])
_input1['PassengerId']
_input1.isnull().sum()
_input1.head()
_input1.HomePlanet = _input1.HomePlanet.fillna('0')
_input1.CryoSleep = _input1.CryoSleep.fillna(0)
_input1.Cabin = _input1.Cabin.fillna(0)
_input1.Destination = _input1.Destination.fillna('0')
_input1.Age = _input1.Age.fillna(0)
_input1.VIP = _input1.VIP.fillna(0)
_input1.RoomService = _input1.RoomService.fillna(0)
_input1.FoodCourt = _input1.FoodCourt.fillna(0)
_input1.ShoppingMall = _input1.ShoppingMall.fillna(0)
_input1.Spa = _input1.Spa.fillna(0)
_input1.VRDeck = _input1.VRDeck.fillna(0)
_input1.Name = _input1.Name.fillna('0')
import pandas as pd
_input1['RoomService'] = _input1['RoomService'].astype(int)
_input1['Age'] = _input1['Age'].astype(int)
_input1['FoodCourt'] = _input1['FoodCourt'].astype(int)
_input1['ShoppingMall'] = _input1['ShoppingMall'].astype(int)
_input1['Spa'] = _input1['Spa'].astype(int)
_input1['VRDeck'] = _input1['VRDeck'].astype(int)
_input1.Destination = _input1.Destination.replace({'TRAPPIST-1e': 'TRAPPIST', '55 Cancri e': 'Cancri', 'PSO J318.5-22': 'PSO'})
_input1['Transported'] = _input1['Transported'].replace(True, 1).replace(False, 0)
import seaborn as sns
sns.stripplot(x='Transported', y='Age', data=_input1, alpha=0.2, jitter=True)
x = _input1.iloc[:, 0:13]
x = _input1.drop(columns=['Cabin', 'Name', 'Transported'])
y = _input1['Transported']
y
_input1.dtypes
x.head()
x.iloc[:, 3].unique()
nom = [1, 3, 2, 5]
num = [0, 6, 7, 8, 9, 10]
bina = [4]
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, Binarizer
from sklearn.compose import make_column_transformer
from sklearn import set_config
transe = make_column_transformer((OneHotEncoder(sparse=False), nom), (PowerTransformer(), num), (Binarizer(), bina), remainder='passthrough')
set_config(display='diagram')
transe
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.25)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
log = LogisticRegression(solver='liblinear')
pipe = make_pipeline(transe, log)
pipe