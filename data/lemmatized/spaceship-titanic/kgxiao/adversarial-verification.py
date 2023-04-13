import pandas as pd
import numpy as np
import os
import catboost as cat
import datetime
import gc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.info()
_input1.isnull().sum()
_input0.isnull().sum()
_input1.head()
_input1 = _input1.fillna(-1, inplace=False)
_input0 = _input0.fillna(-1, inplace=False)
df_train = _input1.copy()
df_test = _input0.copy()
df_train = df_train.drop(columns=['Transported', 'PassengerId', 'Name'], axis=1, inplace=False)
all_columns = df_train.columns.to_list()
df_train['target'] = 1
df_test = df_test.drop(columns=['PassengerId', 'Name'], axis=1, inplace=False)
df_test['target'] = 0
cat_fea = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
df = pd.concat([df_train, df_test]).reset_index(drop=True)
(df_train, df_test) = train_test_split(df, test_size=0.25, random_state=123, stratify=df['target'])

class Catmodel:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def run(self, status=True):
        print('Adversarial Verification …… ……')
        if status:
            model = cat.CatBoostClassifier()