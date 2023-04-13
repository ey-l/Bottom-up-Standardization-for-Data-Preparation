import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.head()
_input1.info()
_input1.nunique()
_input1['Cabin']
cat_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
cont_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1.info()
simp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
simp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
for col in cat_cols:
    _input1[col] = simp_cat.fit_transform(_input1[col].values.reshape(-1, 1))[:, 0]
    _input0[col] = simp_cat.transform(_input0[col].values.reshape(-1, 1))[:, 0]
for col in cont_cols:
    _input1[col] = simp_num.fit_transform(_input1[col].values.reshape(-1, 1))[:, 0]
    _input0[col] = simp_num.transform(_input0[col].values.reshape(-1, 1))[:, 0]
_input1['Deck'] = _input1['Cabin'].apply(lambda x: str(x).split('/')[0])
_input1['Number'] = _input1['Cabin'].apply(lambda x: str(x).split('/')[1]).astype(int)
_input1['Side'] = _input1['Cabin'].apply(lambda x: str(x).split('/')[2])
_input0['Deck'] = _input0['Cabin'].apply(lambda x: str(x).split('/')[0])
_input0['Number'] = _input0['Cabin'].apply(lambda x: str(x).split('/')[1]).astype(int)
_input0['Side'] = _input0['Cabin'].apply(lambda x: str(x).split('/')[2])
cat_cols.append('Deck')
cat_cols.append('Side')
cont_cols.append('Number')

def encoder(df):
    for col in cat_cols:
        le = LabelEncoder()