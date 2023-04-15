import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import h2o
from h2o.automl import H2OAutoML
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
df = pd.read_csv('data/input/spaceship-titanic/train.csv', dtype={'HomePlanet': 'category', 'Cabin': 'string', 'Destination': 'category', 'Age': 'float32', 'RoomService': 'float32', 'FoodCourt': 'float32', 'ShoppingMall': 'float32', 'Spa': 'float32', 'VRDeck': 'float32', 'Transported': 'int'})
df.drop(['Name'], axis=1, inplace=True)
df2 = pd.read_csv('data/input/spaceship-titanic/test.csv', dtype={'HomePlanet': 'category', 'Cabin': 'string', 'Destination': 'category', 'Age': 'float32', 'RoomService': 'float32', 'FoodCourt': 'float32', 'ShoppingMall': 'float32', 'Spa': 'float32', 'VRDeck': 'float32', 'Transported': 'int'})
df2.drop(['Name'], axis=1, inplace=True)
df = df.append(df2)
df.set_index('PassengerId', inplace=True)
df.head()
df.info()
df['Total'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
df
df.isna().sum()
nanrows = df.isna().any(axis=1).sum()
totalrows = df.shape[0]
nanrows / totalrows
df['HomePlanet'].isnull().sum()
sns.countplot(x=df['HomePlanet'])
df['HomePlanet'].fillna('Earth', inplace=True)
df['HomePlanet'].isnull().sum()
sns.countplot(x=df['CryoSleep'])
df['CryoSleep'].isnull().sum()
nancryosleepers = (df['CryoSleep'].isnull() == True) & (df['Total'] == 0)
df.loc[nancryosleepers, 'CryoSleep'] = df.loc[nancryosleepers, 'CryoSleep'].fillna(True)
df['CryoSleep'].isnull().sum()
df['CryoSleep'].fillna(False, inplace=True)
df['CryoSleep'].isnull().sum()
df[(df['CryoSleep'].isnull() == True) & (df['Total'] == 0)]
df['CryoSleep'].isna().value_counts()
df['CryoSleep'].fillna(False, inplace=True)
df['CryoSleep'].isnull().sum()
df[['Deck', 'Number', 'Side']] = df['Cabin'].str.split('/', expand=True)
df.drop('Cabin', axis=1, inplace=True)
df
sns.countplot(x=df['Side'])
df['Side'].fillna('S', inplace=True)
df['Side'].isnull().sum()
df['Deck'].isnull().sum()
sns.countplot(x=df['Deck'])
df['Deck'].fillna('F', inplace=True)
df['Deck'].isnull().sum()
df[['Deck', 'Side']] = df[['Deck', 'Side']].astype('category')
df.info()
df['Number'].isnull().sum()
df['Number'].fillna('0', inplace=True)
df['Number'] = df['Number'].astype('int')
df.info()
plt.hist(x=df['Number'])
sns.countplot(x=df['Destination'])
df['Destination'].isnull().sum()
df['Destination'].fillna('TRAPPIST-1e', inplace=True)
df['Destination'].isnull().sum()
plt.hist(x=df['Age'])
df['Age'].isnull().sum()
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Age'].isnull().sum()
sns.countplot(x=df['VIP'])
df['VIP'].isnull().sum()
df['VIP'].fillna(False, inplace=True)
df['VIP'].isnull().sum()
grafico = px.histogram(df[df['RoomService'] > 0], x='RoomService')
grafico.show()
df['RoomService'].isnull().sum()
grafico = px.histogram(df[df['FoodCourt'] > 0], x='FoodCourt')
grafico.show()
df['FoodCourt'].isnull().sum()
grafico = px.histogram(df[df['ShoppingMall'] > 0], x='ShoppingMall')
grafico.show()
df['ShoppingMall'].isnull().sum()
grafico = px.histogram(df[df['Spa'] > 0], x='Spa')
grafico.show()
df['ShoppingMall'].isnull().sum()
grafico = px.histogram(df[df['VRDeck'] > 0], x='VRDeck')
grafico.show()
df['ShoppingMall'].isnull().sum()
spendings = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
iscryosleep = df['CryoSleep'] == True
awake = df['CryoSleep'] == False
for column in df[spendings]:
    df.loc[iscryosleep, column] = df.loc[iscryosleep, column].fillna(0)
    df.loc[awake, column] = df.loc[awake, column].fillna(df[column].mean())
df['RoomService'].isnull().sum()
df['FoodCourt'].isnull().sum()
df['ShoppingMall'].isnull().sum()
df['Spa'].isnull().sum()
df['VRDeck'].isnull().sum()
grafic = px.treemap(df, path=['CryoSleep', 'RoomService'])
grafic.show()
grafic = px.treemap(df, path=['CryoSleep', 'FoodCourt'])
grafic.show()
grafic = px.treemap(df, path=['CryoSleep', 'ShoppingMall'])
grafic.show()
grafic = px.treemap(df, path=['CryoSleep', 'Spa'])
grafic.show()
grafic = px.treemap(df, path=['CryoSleep', 'VRDeck'])
grafic.show()
df
X = df.drop('Transported', axis=1)
X
Y = df['Transported']
Y
label_encoder_cryo_sleep = LabelEncoder()
label_encoder_vip = LabelEncoder()
label_encoder_side = LabelEncoder()
X.iloc[:, 1] = label_encoder_cryo_sleep.fit_transform(X.iloc[:, 1])
X.iloc[:, 4] = label_encoder_vip.fit_transform(X.iloc[:, 4])
X.iloc[:, -1] = label_encoder_side.fit_transform(X.iloc[:, -1])
X
onehotencoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 2, 11])], remainder='passthrough')
X = pd.DataFrame(onehotencoder.fit_transform(X), columns=onehotencoder.get_feature_names_out(), index=df.index)
X.columns = [column.split('__')[1] for column in X.columns]
X
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=df.index)
X
Y.isna().sum()
df2.shape
X_train = X[:-4277]
X_train
X_test = X[-4277:]
X_test
Y_train = Y[:-4277]
Y_train
Y_test = Y[-4277:]
Y_test
X_train.drop('Total', axis=1, inplace=True)
X_test.drop('Total', axis=1, inplace=True)

def new_model(model, **kwargs):
    new_model = model(**kwargs)