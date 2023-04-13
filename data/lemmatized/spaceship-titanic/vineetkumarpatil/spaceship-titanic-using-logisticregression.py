import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head(3).style.background_gradient(axis=None)
_input0.head(3).style.background_gradient(axis=None)
print('Train data shape :- ' + str(_input1.shape))
print('Test data shape :- ' + str(_input0.shape))
train_transported = _input1['Transported']
df_preprocess = pd.concat((_input1, _input0)).reset_index(drop=True)
df_preprocess = df_preprocess.drop(['Transported'], axis=1, inplace=False)
df_preprocess = df_preprocess.drop(['PassengerId'], axis=1, inplace=False)
df_preprocess = df_preprocess.drop(['Name'], axis=1, inplace=False)
print('Combined data shape :- ' + str(df_preprocess.shape))
df_preprocess.head(3).style.background_gradient(axis=None)
categorical_columns = df_preprocess.dtypes[df_preprocess.dtypes == 'object'].index
print('Categorical Columns :- ')
print(categorical_columns, '\n\n')
numerical_columns = df_preprocess.dtypes[df_preprocess.dtypes != 'object'].index
print('Numerical Columns :- ')
print(numerical_columns, '\n\n')

def get_missing_values_percent(input_dataframe):
    percent_missing = round(input_dataframe.isnull().sum() * 100 / len(input_dataframe), 2)
    missing_value_df = pd.DataFrame({'column_name': input_dataframe.columns, 'percent_missing': percent_missing})
    missing_value_df = missing_value_df[missing_value_df['percent_missing'] > 0]
    missing_value_df = missing_value_df.sort_values(by='percent_missing', ascending=False)
    missing_value_df.set_index('column_name')
    return missing_value_df
missing_value_df = get_missing_values_percent(df_preprocess)
missing_value_df
for col in numerical_columns:
    df_preprocess[col] = df_preprocess[col].fillna(df_preprocess[col].mean(), inplace=False)
for col in categorical_columns:
    df_preprocess[col] = df_preprocess[col].fillna(df_preprocess[col].mode(), inplace=False)
missing_value_df = get_missing_values_percent(df_preprocess)
missing_value_df
print('CryoSleep :- ' + str(df_preprocess['CryoSleep'].mode()), '\n')
print('Cabin :- ' + str(df_preprocess['Cabin'].mode()), '\n')
print('VIP :- ' + str(df_preprocess['VIP'].mode()), '\n')
print('HomePlanet :- ' + str(df_preprocess['HomePlanet'].mode()), '\n')
print('Destination :- ' + str(df_preprocess['Destination'].mode()), '\n')
df_preprocess['CryoSleep'] = df_preprocess['CryoSleep'].fillna('False', inplace=False)
df_preprocess['Cabin'] = df_preprocess['Cabin'].fillna('G/160/P', inplace=False)
df_preprocess['VIP'] = df_preprocess['VIP'].fillna('False', inplace=False)
df_preprocess['HomePlanet'] = df_preprocess['HomePlanet'].fillna('Earth', inplace=False)
df_preprocess['Destination'] = df_preprocess['Destination'].fillna('TRAPPIST-1e', inplace=False)
missing_value_df = get_missing_values_percent(df_preprocess)
missing_value_df
for col in categorical_columns:
    label_encoder_obj = LabelEncoder()