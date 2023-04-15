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
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_train['FoodCourt'].fillna(df_train['FoodCourt'].median(), inplace=True)
df_train['ShoppingMall'].fillna(df_train['ShoppingMall'].median(), inplace=True)
df_train['Spa'].fillna(df_train['Spa'].median(), inplace=True)
df_train['VRDeck'].fillna(df_train['VRDeck'].median(), inplace=True)
df_train['RoomService'].fillna(df_train['RoomService'].median(), inplace=True)
df_train['HomePlanet'].fillna(df_train['HomePlanet'].mode().values[0], inplace=True)
df_train['CryoSleep'].fillna(df_train['CryoSleep'].mode().values[0], inplace=True)
df_train['Destination'].fillna(df_train['Destination'].mode().values[0], inplace=True)
df_train['VIP'].fillna(df_train['VIP'].mode().values[0], inplace=True)
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
df_test['FoodCourt'].fillna(df_test['FoodCourt'].median(), inplace=True)
df_test['ShoppingMall'].fillna(df_test['ShoppingMall'].median(), inplace=True)
df_test['Spa'].fillna(df_test['Spa'].median(), inplace=True)
df_test['VRDeck'].fillna(df_test['VRDeck'].median(), inplace=True)
df_test['RoomService'].fillna(df_test['RoomService'].median(), inplace=True)
df_test['HomePlanet'].fillna(df_test['HomePlanet'].mode().values[0], inplace=True)
df_test['CryoSleep'].fillna(df_test['CryoSleep'].mode().values[0], inplace=True)
df_test['Destination'].fillna(df_test['Destination'].mode().values[0], inplace=True)
df_test['VIP'].fillna(df_test['VIP'].mode().values[0], inplace=True)
df_train['Cabin'].fillna('Z/9999/Z', inplace=True)
df_test['Cabin'].fillna('Z/9999/Z', inplace=True)
df_train['Cabin_deck'] = df_train['Cabin'].apply(lambda x: x.split('/')[0])
df_train['Cabin_number'] = df_train['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
df_train['Cabin_side'] = df_train['Cabin'].apply(lambda x: x.split('/')[2])
df_test['Cabin_deck'] = df_test['Cabin'].apply(lambda x: x.split('/')[0])
df_test['Cabin_number'] = df_test['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
df_test['Cabin_side'] = df_test['Cabin'].apply(lambda x: x.split('/')[2])
df_train.loc[df_train['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
df_train.loc[df_train['Cabin_number'] == 9999, 'Cabin_number'] = np.nan
df_train.loc[df_train['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan
df_test.loc[df_test['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
df_test.loc[df_test['Cabin_number'] == 9999, 'Cabin_number'] = np.nan
df_test.loc[df_test['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan
df_train.drop('Cabin', axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)
from sklearn.model_selection import train_test_split
random_state_val = 42
test_size_val = 0.2
(train, validation) = train_test_split(df_train, test_size=test_size_val, random_state=random_state_val)
y_nm = 'Transported'
df_train_x = train.drop(y_nm, axis=1)
df_train_y = pd.DataFrame(train[y_nm])
df_val_x = validation.drop(y_nm, axis=1)
df_val_y = pd.DataFrame(validation[y_nm])
df_test_x = df_test
num_cols = [col for col in df_train_x.columns if df_train_x[col].dtype in ['float16', 'float32', 'float64']]
cat_cols = [col for col in df_train_x.columns if df_train_x[col].dtype not in ['float16', 'float32', 'float64']]
y = train['Transported'].copy()
for cols in cat_cols:
    enc = TargetEncoder(cols=[cols])
    df_train_x = enc.fit_transform(df_train_x, y)
    df_val_x = enc.transform(df_val_x)
    df_test_x = enc.transform(df_test_x)
scaler = QuantileTransformer()