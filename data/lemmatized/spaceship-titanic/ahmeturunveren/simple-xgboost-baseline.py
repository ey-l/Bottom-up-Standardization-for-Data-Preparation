import seaborn as sns
import sys
import csv
import datetime
import operator
import joblib
import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import norm, skew, probplot
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import StratifiedKFold
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(), inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].median(), inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].median(), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].median(), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].median(), inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].median(), inplace=False)
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(_input1['HomePlanet'].mode().values[0], inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].mode().values[0], inplace=False)
_input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].mode().values[0], inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].mode().values[0], inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median(), inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0['FoodCourt'].median(), inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0['ShoppingMall'].median(), inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(_input0['Spa'].median(), inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0['VRDeck'].median(), inplace=False)
_input0['RoomService'] = _input0['RoomService'].fillna(_input0['RoomService'].median(), inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(_input0['HomePlanet'].mode().values[0], inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(_input0['CryoSleep'].mode().values[0], inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(_input0['Destination'].mode().values[0], inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(_input0['VIP'].mode().values[0], inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna('Z/9999/Z', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna('Z/9999/Z', inplace=False)
_input1['Cabin_deck'] = _input1['Cabin'].apply(lambda x: x.split('/')[0])
_input1['Cabin_number'] = _input1['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
_input1['Cabin_side'] = _input1['Cabin'].apply(lambda x: x.split('/')[2])
_input0['Cabin_deck'] = _input0['Cabin'].apply(lambda x: x.split('/')[0])
_input0['Cabin_number'] = _input0['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
_input0['Cabin_side'] = _input0['Cabin'].apply(lambda x: x.split('/')[2])
_input1.loc[_input1['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
_input1.loc[_input1['Cabin_number'] == 9999, 'Cabin_number'] = np.nan
_input1.loc[_input1['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan
_input0.loc[_input0['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
_input0.loc[_input0['Cabin_number'] == 9999, 'Cabin_number'] = np.nan
_input0.loc[_input0['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
from sklearn.model_selection import train_test_split
random_state_val = 42
test_size_val = 0.2
(train, validation) = train_test_split(_input1, test_size=test_size_val, random_state=random_state_val)
y_nm = 'Transported'
df_train_x = train.drop(y_nm, axis=1)
df_train_y = pd.DataFrame(train[y_nm])
df_val_x = validation.drop(y_nm, axis=1)
df_val_y = pd.DataFrame(validation[y_nm])
df_test_x = _input0
num_cols = [col for col in df_train_x.columns if df_train_x[col].dtype in ['float16', 'float32', 'float64']]
cat_cols = [col for col in df_train_x.columns if df_train_x[col].dtype not in ['float16', 'float32', 'float64']]
y = train['Transported'].copy()
for cols in cat_cols:
    enc = TargetEncoder(cols=[cols])
    df_train_x = enc.fit_transform(df_train_x, y)
    df_val_x = enc.transform(df_val_x)
    df_test_x = enc.transform(df_test_x)
scaler = QuantileTransformer()