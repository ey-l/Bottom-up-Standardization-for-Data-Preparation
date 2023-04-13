import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as nm
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.shape
pas = _input0.PassengerId
_input1.head()
y = _input1.Transported
datas = _input1.drop(['Transported'], axis='columns')
datas = pd.concat([datas, _input0], axis=0)
datas.tail()
datas = datas.drop(['Name', 'Cabin'], axis='columns')
nan_cols = [i for i in datas.columns if datas[i].isnull().any()]
nan_cols
col = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
datas[col] = datas[col].fillna(datas[col].median())
nan_cols = [i for i in datas.columns if datas[i].isnull().any()]
nan_cols
nol = datas.CryoSleep.unique()
nol.size
datas.shape
datas = datas.drop(['PassengerId'], axis='columns')
nan_cols = [i for i in datas.columns if datas[i].isnull().any()]
nan_cols
datas.shape
datas.isna().sum()
datas['VIP'].value_counts()
datas = datas.fillna({'HomePlanet': 'Earth', 'CryoSleep': 'False', 'Destination': 'TRAPPIST-1e', 'VIP': 'False'})
datas.isna().sum()
dataset = pd.get_dummies(datas, drop_first=True)
dataset.head(7)
X = dataset.iloc[:8693, :]
test = dataset.iloc[8693:, :]
test.shape
y.head()
from sklearn.model_selection import train_test_split
(train_X, test_X, train_y, test_y) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=5)
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
model = SVC(C=2)