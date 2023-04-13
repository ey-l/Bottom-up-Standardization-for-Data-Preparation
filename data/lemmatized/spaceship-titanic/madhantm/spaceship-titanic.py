import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head(5)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head(5)
print(' Shape of Train set ', _input1.shape, '\n', 'Shape of Test set', _input0.shape)
_input1.info()
_input1 = _input1.replace({True: 1, False: 0}, inplace=False)
_input1.describe()
_input1.isna().sum()
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1[categorical_cols] = _input1[categorical_cols].fillna(_input1[categorical_cols].mode().iloc[0])
_input1[numeric_cols] = _input1[numeric_cols].fillna(_input1[numeric_cols].mean())
df_cont = _input1[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
plt.figure(figsize=(20, 10), facecolor='Coral')
plotnumber = 1
for column in df_cont:
    if plotnumber <= 6:
        ax = plt.subplot(1, 6, plotnumber)
        sns.boxplot(df_cont[column], color='blueviolet')
        plt.xlabel(column, fontsize=20)
    plotnumber += 1
plt.tight_layout()
df_cont = _input1[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
qua_low = df_cont.quantile(0.015)
qua_high = df_cont.quantile(0.8)
d1 = _input1[(_input1.Age > qua_high.Age) | (_input1.RoomService > qua_high.RoomService) | (_input1.FoodCourt > qua_high.FoodCourt) | (_input1.ShoppingMall > qua_high.ShoppingMall) | (_input1.Spa > qua_high.Spa) | (_input1.VRDeck > qua_high.VRDeck)].index
_input1 = _input1.drop(d1, inplace=False)
_input1.shape
plt.figure(figsize=(20, 10), facecolor='darkorange')
plotnumber = 1
for column in df_cont:
    if plotnumber <= 6:
        ax = plt.subplot(2, 3, plotnumber)
        sns.distplot(df_cont[column], color='darkorchid')
        plt.xlabel(column, fontsize=15)
    plotnumber += 1
plt.tight_layout()
(fig, axs) = plt.subplots(2, 2, figsize=(15, 10), facecolor='plum')
variables = [('Transported', axs[0, 0]), ('VIP', axs[0, 1]), ('HomePlanet', axs[1, 0]), ('CryoSleep', axs[1, 1])]
for (variable, ax) in variables:
    sns.violinplot(data=_input1, y='Age', x=variable, palette='magma', ax=ax)
    ax.set_ylabel('Age', fontsize=15)
    ax.set_xlabel(variable, fontsize=15)
plt.tight_layout()
dfca = _input1[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']]
plt.figure(figsize=(15, 10), facecolor='slateblue')
plotnumber = 1
for columns in dfca:
    if plotnumber <= 4:
        ax = plt.subplot(2, 2, plotnumber)
        sns.countplot(x=dfca[columns], data=_input1, palette='Set1_r')
        plt.xlabel(columns.upper(), fontsize=20)
    plotnumber += 1
plt.tight_layout()
sns.histplot(x=_input1.Age, data=_input1, hue=_input1.Transported, kde=True, palette='autumn')
df_cont1 = df_cont[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
plt.figure(figsize=(20, 10), facecolor='lightcoral')
plotnumber = 1
for columns in df_cont1:
    if plotnumber <= 5:
        ax = plt.subplot(1, 5, plotnumber)
        sns.distplot(x=df_cont1[columns], norm_hist=True, color='midnightblue')
        plt.xlabel(columns.upper(), fontsize=15)
        plt.ylabel('')
    plotnumber += 1
plt.tight_layout()
plt.figure(figsize=(15, 15), facecolor='lightcoral')
sns.heatmap(df_cont.corr(), annot=True, cmap='Paired', linewidths=0.05)
plt.figure(figsize=(12, 10))
corr = df_cont.corr()
clustermap = sns.clustermap(corr, annot=True, cmap='tab20_r', linewidths=0.05)
clustermap.ax_heatmap.set_xlabel('Variables', fontsize=12)
clustermap.ax_heatmap.set_ylabel('Variables', fontsize=12)
df1 = _input1.copy()
dfd = pd.get_dummies(df1)
X = dfd.drop(columns=['Transported'])
y = dfd.Transported
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=100)
lgr = LogisticRegression()