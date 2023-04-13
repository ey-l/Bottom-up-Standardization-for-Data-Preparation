import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier, Pool
from collections import Counter
from tqdm import tqdm
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')

def feature_engineering(df):
    df['pid1'] = df['PassengerId'].apply(lambda x: int(x.split('_')[0]))
    df['pid2'] = df['PassengerId'].apply(lambda x: int(x.split('_')[1]))
    df['pid1_over1k'] = df['pid1'].map(lambda x: int(x / 300))
    if 'Transported' in df.columns:
        df['Transported'] = df['Transported'].map(lambda x: int(x))
    df['Cabin'] = df['Cabin'].map(lambda x: str(x))
    df['cabin1'] = df['Cabin'].apply(lambda x: x.split('/')[0] if x != 'nan' else None)
    df['cabin2'] = df['Cabin'].apply(lambda x: x.split('/')[1] if x != 'nan' else None)
    df['cabin3'] = df['Cabin'].apply(lambda x: x.split('/')[2] if x != 'nan' else None)
    people_per_cabin = dict(Counter(df['Cabin']))
    people_per_cabin.pop('nan')
    df['cabin_mates'] = df['Cabin'].apply(lambda x: people_per_cabin[x] - 1 if x in people_per_cabin else -1)
    df['Age'] = df['Age'] / 100.0
    df['young'] = df['Age'].apply(lambda x: int(x <= 0.2))
    df['mid'] = df['Age'].apply(lambda x: int(0.2 < x <= 0.4))
    df['old'] = df['Age'].apply(lambda x: int(0.4 < x))
    for feature in ['RoomService', 'ShoppingMall', 'FoodCourt', 'Spa', 'VRDeck']:
        df[feature] = np.log(df[feature] + 0.01)
        df[feature.lower() + '_lt_zero'] = df[feature].map(lambda x: int(x < -2))
        df[feature.lower() + '_scaled'] = df[feature] / df['cabin_mates'].map(lambda x: max(x, 1))
    df['foodcourt_thresh'] = df['FoodCourt'].map(lambda x: int(x > 7))
    df['room_thresh'] = df['RoomService'].map(lambda x: int(x > 5))
    df['shop_thresh'] = df['ShoppingMall'].map(lambda x: int(x >= 8))
    df['vr_thresh'] = df['VRDeck'].map(lambda x: int(x >= 8))
    for feat in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']:
        df[feat] = df[feat].fillna('OTHER_UNKNOWN')
    for col in ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name', 'cabin1', 'cabin3']:
        df[col] = df[col].map(lambda x: str(x))
    df['fname'] = df['Name'].apply(lambda x: x.split(' ')[0] if x else '').fillna('')
    df['lname'] = df['Name'].apply(lambda x: x.split(' ')[1] if x and len(x.split(' ')) > 1 else '').fillna('')
    df['fname_len'] = df['fname'].map(lambda x: len(x))
    df['lname_len'] = df['lname'].map(lambda x: len(x))
    df['name_len'] = df['Name'].map(lambda x: len(x))
    return df
_input1 = feature_engineering(_input1)
_input0 = feature_engineering(_input0)
for feat in sorted(_input1.columns):
    if _input1[feat].dtype not in [np.object, np.str] or len(_input1[feat].unique()) < 100:
        plt.hist(_input1[_input1['Transported'] == 0][feat], alpha=0.5)
        plt.hist(_input1[_input1['Transported'] == 1][feat], alpha=0.5)
        plt.title(feat)
        plt.legend(['Not transported', 'Transported'])
plt.hist(_input1[_input1['Transported'] == 0]['pid1'].map(lambda x: int(x / 300)), alpha=0.5, bins=31)
plt.hist(_input1[_input1['Transported'] == 1]['pid1'].map(lambda x: int(x / 300)), alpha=0.5, bins=31)
_input1.columns
categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'cabin1', 'cabin3', 'name_len', 'fname_len', 'lname_len', 'cabin_mates', 'pid1_over1k']
ordinal_cols = ['Cabin', 'cabin2']
numerical_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age', 'pid2', 'cabin2', 'young', 'roomservice_lt_zero', 'shoppingmall_lt_zero', 'foodcourt_lt_zero', 'spa_lt_zero', 'vrdeck_lt_zero']
text_columns = ['fname', 'lname']
categorical_encoder = OneHotEncoder(handle_unknown='ignore')
ordinal_encoder = Pipeline([('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)), ('scale', StandardScaler())])
numerical_pipe = Pipeline([('impute', SimpleImputer(strategy='mean')), ('scale', StandardScaler())])
preprocessing = ColumnTransformer([('cat', categorical_encoder, categorical_columns), ('num', numerical_pipe, numerical_columns), ('ord', ordinal_encoder, ordinal_cols)], verbose_feature_names_out=False)