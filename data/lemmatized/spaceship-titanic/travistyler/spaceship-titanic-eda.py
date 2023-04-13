import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import os, gc, re, warnings
warnings.filterwarnings('ignore')
df = pd.DataFrame(pd.read_csv('data/input/spaceship-titanic/train.csv'))
df
df['CabinDeck'] = df['Cabin'].str[0]
df['CabinNumber'] = df['Cabin'].str.split('/').str[1]
df['CabinSide'] = df['Cabin'].str[-1]
df.head()
(fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10))) = plt.subplots(5, 2, figsize=(15, 30))
df.groupby(['HomePlanet', 'Transported']).size().unstack().plot.bar(title='Home Planet', stacked=True, ax=ax1, rot=0)
df.groupby(['CryoSleep', 'Transported']).size().unstack().plot.bar(title='Cryo-Sleep', stacked=True, ax=ax2, rot=0)
df.groupby(['Destination', 'Transported']).size().unstack().plot.bar(title='Destination', stacked=True, ax=ax3, rot=0)
df.loc[df['Transported'] == False]['Age'].plot.hist(grid=True, title='Age', ax=ax4, alpha=0.5, bins=10)
df.loc[df['Transported'] == True]['Age'].plot.hist(grid=True, ax=ax4, alpha=0.5, bins=10)
df.groupby(['VIP', 'Transported']).size().unstack().plot.bar(title='VIP', stacked=True, ax=ax5, rot=0)
df.loc[df['Transported'] == False]['RoomService'].plot.hist(grid=True, title='Room Service', ax=ax6, alpha=0.5, bins=10, range=[0, 4000])
df.loc[df['Transported'] == True]['RoomService'].plot.hist(grid=True, ax=ax6, alpha=0.5, bins=10)
df.loc[df['Transported'] == False]['ShoppingMall'].plot.hist(grid=True, title='Shopping Mall', ax=ax7, alpha=0.5, bins=10, range=[0, 4000])
df.loc[df['Transported'] == True]['ShoppingMall'].plot.hist(grid=True, ax=ax7, alpha=0.5, bins=10, range=[0, 4000])
df.loc[df['Transported'] == False]['VRDeck'].plot.hist(grid=True, title='VR Deck', ax=ax8, alpha=0.5, bins=10, range=[0, 4000])
df.loc[df['Transported'] == True]['VRDeck'].plot.hist(grid=True, ax=ax8, alpha=0.5, bins=10, range=[0, 4000])
df['Transported'].value_counts().plot.bar(title='Transported', ax=ax9, rot=0)
plt.tight_layout()
(fig, (ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(8, 30))
df.dropna(how='any').groupby(['CabinDeck', 'Transported']).size().unstack().plot.bar(title='Cabin Deck', stacked=True, ax=ax1, rot=0)
df.dropna(how='any').loc[df['Transported'] == False]['CabinNumber'].astype('int').plot.hist(grid=True, title='Cabin Number', ax=ax2, alpha=0.5, bins=10)
df.dropna(how='any').loc[df['Transported'] == True]['CabinNumber'].astype('int').plot.hist(grid=True, ax=ax2, alpha=0.5, bins=10)
df.dropna(how='any').groupby(['CabinSide', 'Transported']).size().unstack().plot.bar(title='Cabin Side', stacked=True, ax=ax3, rot=0)
plt.tight_layout()
plt.figure(figsize=(19, 10))
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.title('Data correlation heatmap')
df['TotalSpent'] = df.RoomService + df.FoodCourt + df.ShoppingMall + df.Spa + df.VRDeck
df.head()
df.loc[df['Transported'] == False]['TotalSpent'].plot.hist(grid=True, title='Total Spent', alpha=0.5, bins=30, range=[0, 12500])
df.loc[df['Transported'] == True]['TotalSpent'].plot.hist(grid=True, alpha=0.5, bins=30, range=[0, 12500])
plt.tight_layout()
df.columns
df2 = df[df['CabinNumber'].notna()]
features = df2[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinDeck', 'CabinNumber', 'CabinSide', 'TotalSpent']]
target = df2['Transported']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(features, target, random_state=42)
print(X_train.shape, y_train.shape)
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
num_features = ['Age', 'RoomService', 'FoodCourt', 'Spa', 'VRDeck', 'TotalSpent', 'CabinNumber']
num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'CabinDeck', 'CabinSide']
cat_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])
clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', MLPClassifier(alpha=1, max_iter=1000))])