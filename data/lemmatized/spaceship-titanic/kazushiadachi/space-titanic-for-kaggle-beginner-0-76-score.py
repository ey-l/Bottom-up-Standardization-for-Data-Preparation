import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
import warnings
warnings.simplefilter('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.head()
_input0.head()
_input2.head()
_input1.info()
_input1.describe()
_input0.info()
_input0.describe()
_input1.isnull().sum()
_input0.isnull().sum()
ax = sns.countplot(x='Transported', data=_input1)
ax.set_title('Transported Counts')
plt.rc('font', size=13)
ax = sns.countplot(x='HomePlanet', data=_input1)
ax.set_title('HomePlanet Counts')
ax = sns.countplot(x='CryoSleep', data=_input1)
ax.set_title('CryoSleep Counts')
ax = sns.countplot(x='Destination', data=_input1)
ax.set_title('Destination Counts')
ax = sns.countplot(x='VIP', data=_input1)
ax.set_title('VIP Counts')
plt.figure(figsize=(8, 4))
ax = sns.histplot(x='Age', data=_input1)
ax.set_title('Age Counts')
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df_ID = _input0['PassengerId']
imputer_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='median')