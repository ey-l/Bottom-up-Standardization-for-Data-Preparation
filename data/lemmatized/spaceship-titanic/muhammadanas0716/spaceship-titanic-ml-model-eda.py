import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
print('Libraries Imported')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Data Loaded Successfully')
_input1.head()
_input1.info()
_input1.describe()
_input1.isna().sum()
(fig, ax) = plt.subplots(2, 1, figsize=(10, 8), dpi=120)
_input1.isna().sum().plot(kind='barh', color=['#003049', '#d62828'], ax=ax[0])
_input0.isna().sum().plot(kind='barh', color=['#000000', '#14213d'], ax=ax[1])
_input1.select_dtypes('Float64').columns
px.histogram(data_frame=_input1, x='Age')
px.histogram(data_frame=_input1, x='RoomService')
px.histogram(data_frame=_input1, x='FoodCourt')
px.histogram(data_frame=_input1, x='ShoppingMall')
px.histogram(data_frame=_input1, x='Spa')
px.histogram(data_frame=_input1, x='VRDeck')
_input1.select_dtypes('O').columns

def plot_value_counts(column_name):
    """
    Pass in the Categorical column name and this function will plot the value counts
    
    **Parameters**
        * column_name: Categorical Column Name
    """
    plt.figure(figsize=(8, 5), dpi=120)
    _input1[column_name].value_counts().plot(kind='bar', color=['#52796f', '#354f52']).set(title=f'Value Counts of the {column_name} Column')
    plt.xticks(rotation=0)
plot_value_counts('HomePlanet')
plot_value_counts('CryoSleep')
plot_value_counts('Destination')
plot_value_counts('VIP')
plot_value_counts('Transported')
(fig, ax) = plt.subplots(1, 2, figsize=(10, 7), dpi=120)
sns.countplot(x=_input1['Transported'], hue=_input1['HomePlanet'], ax=ax[0]).set(title='HomePlanet')
sns.countplot(x=_input1['Transported'], hue=_input1['Destination']).set(title='Destination')
_input1.columns
sns.countplot(x=_input1['Transported'], hue=_input1['VIP']).set(title='VIP')
temp_df = _input1.copy()
test_df_copy = _input0.copy()
temp_df
temp_df.select_dtypes('float').isna().sum()
temp_df.select_dtypes('float').mode()
imputer_startegy = SimpleImputer(strategy='most_frequent')
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
object_features = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']
imputer = ColumnTransformer([('num_imputer', imputer_startegy, numerical_features), ('object_imputer', imputer_startegy, object_features)])
temp_df = imputer.fit_transform(temp_df)
test_df_copy = imputer.fit_transform(_input0)
temp_df = pd.DataFrame(temp_df, columns=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name'])
test_df_copy = pd.DataFrame(test_df_copy, columns=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name'])
test_df_copy.isna().sum()
transported = _input1.Transported
temp_df['Transported'] = transported
temp_df.head()
temp_df.Cabin
temp_df = temp_df[['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Transported']]
temp_df.Cabin.str.contains('P').value_counts()
temp_df.loc[temp_df['Cabin'].str.contains('P'), 'Spot'] = 'Pilot' if True else 'StarBoard'
temp_df.loc[temp_df['Cabin'].str.contains('S'), 'Spot'] = 'StarBoard' if True else 'Pilot'
test_df_copy.loc[temp_df['Cabin'].str.contains('P'), 'Spot'] = 'Pilot' if True else 'StarBoard'
test_df_copy.loc[temp_df['Cabin'].str.contains('S'), 'Spot'] = 'StarBoard' if True else 'Pilot'
temp_df = temp_df.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=False)
test_df_copy = test_df_copy.drop(['Name', 'Cabin'], axis=1, inplace=False)
num_cols = _input1.select_dtypes('float').columns
num_cols
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in num_cols:
    temp_df[col] = temp_df[col].astype('float64')
    test_df_copy[col] = test_df_copy[col].astype('float64')
test_df_copy
categorical_cols = temp_df.select_dtypes('object').columns.tolist()
numeric_cols = temp_df.select_dtypes('float').columns.tolist()
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')