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
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
data.columns = [col.lower() for col in data.columns]
data
data.info()
data['group_id'] = data['passengerid'].apply(lambda x: x.split('_')[0])
data['person_id'] = data['passengerid'].apply(lambda x: x.split('_')[1])
data['cabin'] = data['cabin'].replace(to_replace=[np.nan], value=['nan/nan/nan'])
data['deck'] = data['cabin'].apply(lambda x: x.split('/')[0])
data['cabin_num'] = data['cabin'].apply(lambda x: x.split('/')[1])
data['cabin_side'] = data['cabin'].apply(lambda x: x.split('/')[2])
data.drop(['cabin'], axis=1, inplace=True)

def plot_pie(col_name):
    fig = px.pie(names=data[col_name].value_counts().index, values=data[col_name].value_counts().values)
    fig.show()
plot_pie('homeplanet')
plot_pie('cryosleep')
plot_pie('destination')
px.histogram(data['age'])
plot_pie('vip')
data
data.groupby('vip')['transported'].value_counts().unstack().plot(kind='bar')
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
    (data, imputer_dict[col]) = impute_col(data, col)
for col in mean:
    (data, imputer_dict[col]) = impute_col(data, col, strat='mean')
for col in constant:
    (data, imputer_dict[col]) = impute_col(data, col, strat='constant')
for col in most_freq_art:
    (data, imputer_dict[col]) = impute_col(data, col, strat='most_frequent', missing_values='nan')
encode_dict = {}

def encode_col(df, cols):
    label_encoder = LabelEncoder()
    df[cols] = label_encoder.fit_transform(df[cols])
    return (df, label_encoder)
for col in most_frequent + ['transported', 'deck', 'cabin_side']:
    (data, encode_dict[col]) = encode_col(data, col)
data
data.info()
to_int = ['group_id', 'person_id', 'cabin_num']
data[to_int] = data[to_int].astype(int)
passenger_id = data['passengerid']
data.drop(['passengerid', 'name'], axis=1, inplace=True)
data.corr()
data.drop(['group_id', 'age', 'vip'], axis=1, inplace=True)
x = data.loc[:, ~data.columns.isin(['transported'])]
y = data['transported']
scaler = StandardScaler()
x = scaler.fit_transform(x)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=177013)
lr = GradientBoostingClassifier(n_estimators=400, max_depth=5, learning_rate=0.1, verbose=1, random_state=177013)