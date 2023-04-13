import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.columns = [col.lower() for col in _input1.columns]
_input1
_input1.info()
_input1['group_id'] = _input1['passengerid'].apply(lambda x: x.split('_')[0])
_input1['person_id'] = _input1['passengerid'].apply(lambda x: x.split('_')[1])
_input1['cabin'] = _input1['cabin'].replace(to_replace=[np.nan], value=['nan/nan/nan'])
_input1['deck'] = _input1['cabin'].apply(lambda x: x.split('/')[0])
_input1['cabin_num'] = _input1['cabin'].apply(lambda x: x.split('/')[1])
_input1['cabin_side'] = _input1['cabin'].apply(lambda x: x.split('/')[2])
_input1 = _input1.drop(['cabin'], axis=1, inplace=False)

def plot_pie(col_name):
    fig = px.pie(names=_input1[col_name].value_counts().index, values=_input1[col_name].value_counts().values)
    fig.show()
plot_pie('homeplanet')
plot_pie('cryosleep')
plot_pie('destination')
px.histogram(_input1['age'])
plot_pie('vip')
_input1
_input1.groupby('vip')['transported'].value_counts().unstack().plot(kind='bar')
imputer_dict = {}

def impute_col(df, col_name, strat='most_frequent', fill_value='UNK', missing_values=np.nan):
    if strat == 'constant':
        imputer = SimpleImputer(strategy=strat, fill_value=fill_value, missing_values=missing_values)
    else:
        imputer = SimpleImputer(strategy=strat, missing_values=missing_values)
    df[col_name] = imputer.fit_transform(df[col_name].values.reshape(-1, 1))
    return (df, imputer)
most_frequent = ['homeplanet', 'cryosleep', 'destination', 'vip']
mean = ['age', 'roomservice', 'foodcourt', 'shoppingmall', 'spa', 'vrdeck']
constant = ['name']
most_freq_art = ['deck', 'cabin_num', 'cabin_side']
for col in most_frequent:
    (_input1, imputer_dict[col]) = impute_col(_input1, col)
for col in mean:
    (_input1, imputer_dict[col]) = impute_col(_input1, col, strat='mean')
for col in constant:
    (_input1, imputer_dict[col]) = impute_col(_input1, col, strat='constant')
for col in most_freq_art:
    (_input1, imputer_dict[col]) = impute_col(_input1, col, strat='most_frequent', missing_values='nan')
encode_dict = {}

def encode_col(df, cols):
    label_encoder = LabelEncoder()
    df[cols] = label_encoder.fit_transform(df[cols])
    return (df, label_encoder)
for col in most_frequent + ['transported', 'deck', 'cabin_side']:
    (_input1, encode_dict[col]) = encode_col(_input1, col)
_input1
_input1.info()
to_int = ['group_id', 'person_id', 'cabin_num']
_input1[to_int] = _input1[to_int].astype(int)
passenger_id = _input1['passengerid']
_input1 = _input1.drop(['passengerid', 'name'], axis=1, inplace=False)
_input1.corr()
_input1 = _input1.drop(['group_id', 'age', 'vip'], axis=1, inplace=False)
x = _input1.loc[:, ~_input1.columns.isin(['transported'])]
y = _input1['transported']
scaler = StandardScaler()
x = scaler.fit_transform(x)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=177013)
lr = GradientBoostingClassifier(n_estimators=400, max_depth=5, learning_rate=0.1, verbose=1, random_state=177013)