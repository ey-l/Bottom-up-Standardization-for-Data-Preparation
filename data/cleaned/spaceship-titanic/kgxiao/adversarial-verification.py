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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.info()
train.isnull().sum()
test.isnull().sum()
train.head()
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)
df_train = train.copy()
df_test = test.copy()
df_train.drop(columns=['Transported', 'PassengerId', 'Name'], axis=1, inplace=True)
all_columns = df_train.columns.to_list()
df_train['target'] = 1
df_test.drop(columns=['PassengerId', 'Name'], axis=1, inplace=True)
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