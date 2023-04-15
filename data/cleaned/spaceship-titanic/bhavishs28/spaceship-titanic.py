import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
data
data.isnull().sum()
import seaborn as sns
sns.heatmap(data.isnull())
from sklearn import preprocessing
Lb = preprocessing.LabelEncoder()
data['PassengerId'] = Lb.fit_transform(data['PassengerId'])
data['PassengerId']
data.isnull().sum()
data.head()
data.HomePlanet = data.HomePlanet.fillna('0')
data.CryoSleep = data.CryoSleep.fillna(0)
data.Cabin = data.Cabin.fillna(0)
data.Destination = data.Destination.fillna('0')
data.Age = data.Age.fillna(0)
data.VIP = data.VIP.fillna(0)
data.RoomService = data.RoomService.fillna(0)
data.FoodCourt = data.FoodCourt.fillna(0)
data.ShoppingMall = data.ShoppingMall.fillna(0)
data.Spa = data.Spa.fillna(0)
data.VRDeck = data.VRDeck.fillna(0)
data.Name = data.Name.fillna('0')
import pandas as pd
data['RoomService'] = data['RoomService'].astype(int)
data['Age'] = data['Age'].astype(int)
data['FoodCourt'] = data['FoodCourt'].astype(int)
data['ShoppingMall'] = data['ShoppingMall'].astype(int)
data['Spa'] = data['Spa'].astype(int)
data['VRDeck'] = data['VRDeck'].astype(int)
data.Destination = data.Destination.replace({'TRAPPIST-1e': 'TRAPPIST', '55 Cancri e': 'Cancri', 'PSO J318.5-22': 'PSO'})
data['Transported'] = data['Transported'].replace(True, 1).replace(False, 0)
import seaborn as sns
sns.stripplot(x='Transported', y='Age', data=data, alpha=0.2, jitter=True)
x = data.iloc[:, 0:13]
x = data.drop(columns=['Cabin', 'Name', 'Transported'])
y = data['Transported']
y
data.dtypes
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