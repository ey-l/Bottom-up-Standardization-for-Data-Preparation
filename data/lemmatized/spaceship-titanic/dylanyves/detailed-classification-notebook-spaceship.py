import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
import missingno as msno
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.shape
_input1.describe()
_input1.info()
print(f'Nan values:\n\n{_input1.isna().sum()}')
msno.matrix(_input1)
print(f'Total duplciate data: {_input1.duplicated().sum()}')
corr = _input1.corr()
plt.figure(figsize=(15, 8))
sns.heatmap(corr, annot=True)
from sklearn.impute import SimpleImputer

class FE:

    def __init__(self, df):
        self.df = df

    def add_columns(self):
        self.df['Deck'] = self.df.Cabin.apply(lambda x: str(x)[0])
        self.df['TotalBill'] = self.df.RoomService + self.df.ShoppingMall + self.df.Spa + self.df.VRDeck
        return self.df

    def fill_na_object(self):
        columns = self.df.select_dtypes(include='object')
        for column in columns:
            val = self.df[column].value_counts().index[0]
            self.df[column] = self.df[column].fillna(val, inplace=False)
        return self.df

    def fill_na_int(self):
        self.df.Age = self.df.Age.fillna(self.df.Age.mean(), inplace=False)
        imputer = SimpleImputer()
        columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        result = imputer.fit_transform(self.df[columns])
        self.df[columns] = result
        return self.df

    def run_all(self):
        self.fill_na_object()
        self.fill_na_int()
        self.add_columns()
        try:
            return self.df[['PassengerId', 'Name', 'Age', 'HomePlanet', 'Destination', 'CryoSleep', 'Cabin', 'Deck', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalBill', 'Transported']]
        except:
            return self.df
fe = FE(_input1.copy())
cleaned_df = fe.run_all()
print(f'Nan values:\n\n{cleaned_df.isna().sum()}')
home_planets = cleaned_df.HomePlanet.value_counts().index
home_planets_count = cleaned_df.HomePlanet.value_counts().values
destinations = cleaned_df.Destination.value_counts().index
destinations_count = cleaned_df.Destination.value_counts().values
decks = cleaned_df.Deck.value_counts().index
decks_count = cleaned_df.Deck.value_counts().values
vip = cleaned_df.VIP.value_counts().index
vip_count = cleaned_df.VIP.value_counts().values
cryo_sleep = cleaned_df.CryoSleep.value_counts().index
cryo_sleep_count = cleaned_df.CryoSleep.value_counts().values
transported = cleaned_df.Transported.value_counts().index
transported_count = cleaned_df.Transported.value_counts().values
(fig, ax) = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
ax[0, 0].pie(home_planets_count, labels=home_planets, startangle=90, autopct='%1.1f%%', shadow=True, wedgeprops={'edgecolor': 'black'}, explode=[0.1, 0, 0], colors=['#d8f9d8', '#f88888', '#aee'])
ax[0, 0].set_title('Home Planets', fontsize=16)
ax[0, 1].pie(destinations_count, labels=destinations, startangle=90, autopct='%1.1f%%', shadow=True, wedgeprops={'edgecolor': 'black'}, explode=[0.1, 0, 0], colors=['#d8f9d8', '#f88888', '#aee'])
ax[0, 1].set_title('Destinations', fontsize=16)
ax[0, 2].pie(decks_count, labels=decks, startangle=220, autopct='%1.1f%%', shadow=True, wedgeprops={'edgecolor': 'black'}, explode=[0, 0.1, 0, 0, 0, 0, 0, 0], colors=['#ececff', '#beb', '#d8f9d8', '#f88888', '#ffdddd', '#aee', '#cff4f4'])
ax[0, 2].set_title('Decks', fontsize=16)
ax[1, 0].pie(vip_count, labels=vip, startangle=90, autopct='%1.1f%%', shadow=True, wedgeprops={'edgecolor': 'black'}, explode=[0, 0.1], colors=['#d8f9d8', '#f88888'])
ax[1, 0].set_title('VIP', fontsize=16)
ax[1, 1].pie(cryo_sleep_count, labels=cryo_sleep, startangle=90, autopct='%1.1f%%', shadow=True, wedgeprops={'edgecolor': 'black'}, explode=[0, 0.1], colors=['#d8f9d8', '#f88888'])
ax[1, 1].set_title('CryoSleep', fontsize=16)
ax[1, 2].pie(transported_count, labels=transported, startangle=90, autopct='%1.1f%%', shadow=True, wedgeprops={'edgecolor': 'black'}, explode=[0, 0.1], colors=['#d8f9d8', '#f88888'])
ax[1, 2].set_title('Transported', fontsize=16)
plt.tight_layout()
fig = px.histogram(cleaned_df, x='Age', marginal='box', title='Age Distribution', color='HomePlanet')
fig.show()
fig = px.box(cleaned_df, x='TotalBill', title='TotalBill Distribution', color='HomePlanet')
fig.show()
planets_bills = cleaned_df.groupby('HomePlanet')['TotalBill'].median().index.tolist()
average_bills = cleaned_df.groupby('HomePlanet')['TotalBill'].median().values.tolist()
planets_age = cleaned_df.groupby('HomePlanet')['Age'].median().index.tolist()
average_age = cleaned_df.groupby('HomePlanet')['Age'].median().values.tolist()
fig = make_subplots(rows=1, cols=2, subplot_titles=('Average Spending', 'Average Age'))
fig.add_trace(go.Bar(x=planets_bills, y=average_bills), row=1, col=1)
fig.add_trace(go.Bar(x=planets_age, y=average_age), row=1, col=2)
fig = px.box(cleaned_df, x='Age', color='Transported', title='Age Distribution')
fig.show()
fig = px.box(cleaned_df, x='TotalBill', color='Transported', title='Total Bill Distribution')
fig.show()
features = ['Age', 'HomePlanet', 'Destination', 'CryoSleep', 'Deck', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalBill']
X = cleaned_df[features]
y = cleaned_df.Transported
try:
    X = pd.get_dummies(X, columns=['HomePlanet', 'Destination', 'CryoSleep', 'Deck', 'VIP'])
except:
    pass
X.head()
from sklearn.model_selection import train_test_split
(train_x, test_x, train_y, test_y) = train_test_split(X, y, test_size=0.25, random_state=1)
print(f'Train size: {train_x.shape}')
print(f'Test size: {test_x.shape}')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()