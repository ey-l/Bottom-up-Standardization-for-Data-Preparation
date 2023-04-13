import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1['cabin_side'] = _input1.apply(lambda x: str(x['Cabin'])[-1:], axis=1)
_input1
_input1.info()
_input1.describe()
fig = px.histogram(_input1, x='HomePlanet', color='Destination')
fig.show()
fig = px.histogram(_input1, x='CryoSleep', color='Transported')
fig.show()
train_data_age = _input1[(_input1['Age'] > 20) & (_input1['Age'] < 40)]
train_data_bar = train_data_age.groupby(['Age']).sum(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
train_data_bar['cabin_side'] = train_data_bar.index
train_data_bar
px.histogram(train_data_bar)
train_area = train_data_bar[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
train_area.plot(kind='barh', title='Amenities ')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
_input1
(_input1, val_data) = train_test_split(_input1, test_size=0.2, random_state=42)
print('size of training data: ', _input1.shape)
print('size of validation data: ', val_data.shape)
input_cols = _input1.columns[1:-2].tolist() + ['cabin_side']
input_cols.remove('Cabin')
input_cols.remove('Name')
target_cols = 'Transported'
print('input columns : ', input_cols)
print('target columns : ', target_cols)
train_inputs = _input1[input_cols].copy()
train_targets = _input1[target_cols].copy()
val_inputs = val_data[input_cols].copy()
val_targets = val_data[target_cols].copy()
train_inputs.info()
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
cat_cols = train_inputs.select_dtypes(include='object').columns
train_inputs[cat_cols]
train_inputs[numeric_cols]
train_inputs[numeric_cols].isna().sum()
imputer = SimpleImputer(strategy='mean')