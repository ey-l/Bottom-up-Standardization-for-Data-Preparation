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
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head(5)
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test.head(5)
print(' Shape of Train set ', df.shape, '\n', 'Shape of Test set', df_test.shape)
df.info()
df.replace({True: 1, False: 0}, inplace=True)
df.describe()
df.isna().sum()
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df_cont = df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
plt.figure(figsize=(20, 10), facecolor='Coral')
plotnumber = 1
for column in df_cont:
    if plotnumber <= 6:
        ax = plt.subplot(1, 6, plotnumber)
        sns.boxplot(df_cont[column], color='blueviolet')
        plt.xlabel(column, fontsize=20)
    plotnumber += 1
plt.tight_layout()

df_cont = df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
qua_low = df_cont.quantile(0.015)
qua_high = df_cont.quantile(0.8)
d1 = df[(df.Age > qua_high.Age) | (df.RoomService > qua_high.RoomService) | (df.FoodCourt > qua_high.FoodCourt) | (df.ShoppingMall > qua_high.ShoppingMall) | (df.Spa > qua_high.Spa) | (df.VRDeck > qua_high.VRDeck)].index
df.drop(d1, inplace=True)
df.shape
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
    sns.violinplot(data=df, y='Age', x=variable, palette='magma', ax=ax)
    ax.set_ylabel('Age', fontsize=15)
    ax.set_xlabel(variable, fontsize=15)
plt.tight_layout()
dfca = df[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']]
plt.figure(figsize=(15, 10), facecolor='slateblue')
plotnumber = 1
for columns in dfca:
    if plotnumber <= 4:
        ax = plt.subplot(2, 2, plotnumber)
        sns.countplot(x=dfca[columns], data=df, palette='Set1_r')
        plt.xlabel(columns.upper(), fontsize=20)
    plotnumber += 1
plt.tight_layout()
sns.histplot(x=df.Age, data=df, hue=df.Transported, kde=True, palette='autumn')
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
df1 = df.copy()
dfd = pd.get_dummies(df1)
X = dfd.drop(columns=['Transported'])
y = dfd.Transported
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=100)
lgr = LogisticRegression()