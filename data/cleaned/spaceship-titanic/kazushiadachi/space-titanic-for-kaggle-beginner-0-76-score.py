import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
import warnings
warnings.simplefilter('ignore')
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
sub = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train_data.head()
test_data.head()
sub.head()
train_data.info()
train_data.describe()
test_data.info()
test_data.describe()
train_data.isnull().sum()
test_data.isnull().sum()
ax = sns.countplot(x='Transported', data=train_data)
ax.set_title('Transported Counts')
plt.rc('font', size=13)
ax = sns.countplot(x='HomePlanet', data=train_data)
ax.set_title('HomePlanet Counts')
ax = sns.countplot(x='CryoSleep', data=train_data)
ax.set_title('CryoSleep Counts')
ax = sns.countplot(x='Destination', data=train_data)
ax.set_title('Destination Counts')
ax = sns.countplot(x='VIP', data=train_data)
ax.set_title('VIP Counts')
plt.figure(figsize=(8, 4))
ax = sns.histplot(x='Age', data=train_data)
ax.set_title('Age Counts')
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df_ID = test_df['PassengerId']
imputer_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='median')