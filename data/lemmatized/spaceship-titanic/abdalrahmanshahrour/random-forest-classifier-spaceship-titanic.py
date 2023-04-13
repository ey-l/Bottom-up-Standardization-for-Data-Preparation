import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.isna().sum()
_input0.isna().sum()
transported_count = _input1['Transported'].value_counts()
(fig, ax) = plt.subplots(figsize=(5, 5))
bars = ax.barh(transported_count.index.astype(str), transported_count.tolist())
ax.bar_label(bars)
_input1.corr()
fig = px.imshow(_input1.corr(), text_auto=True)
fig.show()
del _input1['Name']
del _input1['PassengerId']
del _input1['Cabin']
del _input0['Name']
del _input0['PassengerId']
del _input0['Cabin']
_input1.info()
df = px.data.tips()
fig = px.histogram(_input1, x='HomePlanet')
fig.show()
fig = px.histogram(_input0, x='HomePlanet')
fig.show()
fig = px.histogram(_input1, x='CryoSleep')
fig.show()
fig = px.histogram(_input0, x='CryoSleep')
fig.show()
fig = px.histogram(_input1, x='Destination')
fig.show()
fig = px.histogram(_input0, x='Destination')
fig.show()
fig = px.histogram(_input1, x='VIP')
fig.show()
fig = px.histogram(_input0, x='VIP')
fig.show()
fig = px.histogram(_input1, x='Transported')
fig.show()

def ProcessNum(df):
    num_data = df.select_dtypes(['float64'])
    num_col = list(num_data.columns)
    dict_num = {i: num_col[i] for i in range(len(num_col))}
    num_data.head()
    imputer = SimpleImputer(strategy='mean')
    d = imputer.fit_transform(num_data)
    temp = pd.DataFrame(d)
    temp = temp.rename(columns=dict_num)
    return temp

def ProcessObj(df):
    obj_data = df.select_dtypes(['object'])
    obj_col = list(obj_data.columns)
    for col in list(obj_data.columns):
        obj_data[col] = obj_data[col].fillna(obj_data[col].mode()[0])
        pass
    z = obj_data.columns
    for i in z:
        un = obj_data[i].unique()
        ran = range(1, len(un) + 1)
        obj_data = obj_data.replace(dict(zip(un, ran)), inplace=False)
    obj_data = pd.get_dummies(obj_data, columns=['HomePlanet', 'Destination'], prefix=['HomePlanet', 'Destination'])
    return obj_data

def ProcessBool(df):
    bool_data = df.select_dtypes(['bool'])
    col = bool_data.columns
    for i in col:
        bool_data[i] = LabelEncoder().fit_transform(bool_data[i])
    return bool_data
test_num = ProcessNum(_input0)
train_num = ProcessNum(_input1)
test_cat = ProcessObj(_input0)
train_cat = ProcessObj(_input1)
train_bool = ProcessBool(_input1)
train_data_process = pd.concat([train_num, train_cat, train_bool], axis=1)
test_data_process = pd.concat([test_num, test_cat], axis=1)
train_data_process.sample(2)
train_data_process.info()
train_data_process.isna().sum()
test_data_process.sample(2)
test_data_process.info()
test_data_process.isna().sum()
features = np.array(train_data_process.drop('Transported', axis=1))
targets = np.array(train_data_process['Transported'])
(x_train, x_val, y_train, y_val) = train_test_split(features, targets, test_size=0.2, random_state=0)
rf = RandomForestClassifier()
RF_grid = {'n_estimators': [50, 100, 150, 200], 'max_depth': [4, 6, 8, 10, 12]}
grid = GridSearchCV(rf, RF_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)