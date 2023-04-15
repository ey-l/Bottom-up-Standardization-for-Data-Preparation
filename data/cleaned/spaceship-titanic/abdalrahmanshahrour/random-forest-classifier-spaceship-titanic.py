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
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head()
df_train.isna().sum()
df_test.isna().sum()
transported_count = df_train['Transported'].value_counts()
(fig, ax) = plt.subplots(figsize=(5, 5))
bars = ax.barh(transported_count.index.astype(str), transported_count.tolist())
ax.bar_label(bars)
df_train.corr()
fig = px.imshow(df_train.corr(), text_auto=True)
fig.show()
del df_train['Name']
del df_train['PassengerId']
del df_train['Cabin']
del df_test['Name']
del df_test['PassengerId']
del df_test['Cabin']
df_train.info()
df = px.data.tips()
fig = px.histogram(df_train, x='HomePlanet')
fig.show()
fig = px.histogram(df_test, x='HomePlanet')
fig.show()
fig = px.histogram(df_train, x='CryoSleep')
fig.show()
fig = px.histogram(df_test, x='CryoSleep')
fig.show()
fig = px.histogram(df_train, x='Destination')
fig.show()
fig = px.histogram(df_test, x='Destination')
fig.show()
fig = px.histogram(df_train, x='VIP')
fig.show()
fig = px.histogram(df_test, x='VIP')
fig.show()
fig = px.histogram(df_train, x='Transported')
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
        obj_data.replace(dict(zip(un, ran)), inplace=True)
    obj_data = pd.get_dummies(obj_data, columns=['HomePlanet', 'Destination'], prefix=['HomePlanet', 'Destination'])
    return obj_data

def ProcessBool(df):
    bool_data = df.select_dtypes(['bool'])
    col = bool_data.columns
    for i in col:
        bool_data[i] = LabelEncoder().fit_transform(bool_data[i])
    return bool_data
test_num = ProcessNum(df_test)
train_num = ProcessNum(df_train)
test_cat = ProcessObj(df_test)
train_cat = ProcessObj(df_train)
train_bool = ProcessBool(df_train)
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