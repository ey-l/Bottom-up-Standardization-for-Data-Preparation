import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.metrics import classification_report
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
RGB = (139 / 255, 10 / 255, 80 / 255)
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(f'Train dataframe has {train_df.shape[0]} training examples and {train_df.shape[1]} features')
print(f'Test dataframe has {test_df.shape[0]} Testing examples and {test_df.shape[1]} features')
print('Train dataframe information')
train_df.info()
print('\n Test dataframe information')
test_df.info()

def Null_test(df, plot=False):
    print(df.isnull().sum())
    if plot:
        msno.matrix(df, color=RGB).set_title('Null values', fontsize=20)
Null_test(train_df, plot=True)
Null_test(test_df, plot=True)
Expenses = ['VRDeck', 'Spa', 'ShoppingMall', 'FoodCourt', 'RoomService']

def FeatureEngineering(dataframe):
    df = dataframe.copy()
    df[['CabinDeck', 'CabinNum', 'CabinSide']] = df['Cabin'].str.split(pat='/', expand=True)
    df[['PassengerIdGroup', 'PassengerIdNum']] = df['PassengerId'].str.split(pat='_', expand=True)
    df['CabinNum'] = df['CabinNum'].astype(str).astype(float)
    df['HomePlanet'].fillna('Earth', inplace=True)
    df['Destination'].fillna('TRAPPIST-1e', inplace=True)
    df['CabinSide'].fillna('P', inplace=True)
    df['CabinDeck'].fillna('T', inplace=True)
    df['CryoSleep'].fillna('False', inplace=True)
    df['CryoSleep'] = df['CryoSleep'].astype(bool).astype(int)
    df['VIP'] = df['VIP'].astype(bool).astype(int)
    df['PassengerIdGroup'] = df['PassengerIdGroup'].astype(str).astype('category')
    df.replace({'Transported': {False: 0, True: 1}}, inplace=True)
    for col in Expenses:
        df[col].fillna(0, inplace=True)
    df['TotalCosts'] = df[Expenses].sum(axis=1)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['CabinNum'].fillna(df['CabinNum'].mean(), inplace=True)
    df.drop(['Cabin', 'Name'], axis=1, inplace=True)
    return df
train_df_new = FeatureEngineering(train_df)
test_df_new = FeatureEngineering(test_df)
print('Train dataframe information')
train_df_new.info()
print('\n Test dataframe information')
test_df_new.info()
Null_test(train_df_new, plot=True)
Null_test(test_df_new, plot=True)
colorPal = ['#780060', '#4CAF50']
piee = train_df_new['Transported'].value_counts()
mylabels = ('True', 'False')
myexplode = [0.2, 0]
(fig, ax) = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect='equal'))
(patches, texts, autotexts) = ax.pie(piee, autopct='%1.1f%%', explode=myexplode, labels=mylabels, shadow=True, colors=colorPal, startangle=90, radius=1.2)
ax.legend(patches, mylabels, title='Transported: ', loc='lower left', fontsize=15, bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=15, weight='bold')
plt.setp(texts, size=15, weight='bold')
ax.set_title('Target distribution', fontsize=15, weight='bold')

corr = train_df_new.corr()
plt.figure(figsize=(12, 12))
sns.set(font_scale=0.8)
sns.heatmap(corr, square=True, cmap='PRGn', linewidths=2, annot=True, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix', fontsize=16, pad=30)

sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor': 'black'})
colorPal = ['#780060', '#4CAF50']
(fig, ax) = plt.subplots(nrows=3, ncols=2, figsize=(15, 10), squeeze=False)
sns.countplot(ax=ax[0, 0], x='HomePlanet', hue='Transported', data=train_df_new, palette=colorPal)
sns.countplot(ax=ax[0, 1], x='Destination', hue='Transported', data=train_df_new, palette=colorPal)
sns.countplot(ax=ax[1, 0], x='CryoSleep', hue='Transported', data=train_df_new, palette=colorPal)
sns.countplot(ax=ax[1, 1], x='CabinDeck', hue='Transported', data=train_df_new, palette=colorPal)
sns.countplot(ax=ax[2, 0], x='CabinSide', hue='Transported', data=train_df_new, palette=colorPal)
sns.countplot(ax=ax[2, 1], x='VIP', hue='Transported', data=train_df_new, palette=colorPal)
sns.histplot(data=train_df_new, x='CabinNum', hue='Transported', palette=colorPal, binwidth=80)
sns.catplot(data=train_df_new, x='HomePlanet', y='Age', hue='Transported', kind='violin', facet_kws=dict(despine=False), aspect=2, palette=colorPal)
object_cols = ['HomePlanet', 'Destination', 'CabinSide', 'CabinDeck']
train_df_new = pd.get_dummies(train_df_new, columns=object_cols, dummy_na=False)
test_df_new = pd.get_dummies(test_df_new, columns=object_cols, dummy_na=False)
train_df_new.info()
from sklearn.preprocessing import StandardScaler

def get_xy(dataframe, y_label, x_label=None):
    if x_label is None:
        X = dataframe[[c for c in dataframe.columns if c != y_label]].values
    elif len(x_label) == 1:
        X = dataframe[x_label[0]].values.reshape(-1, 1)
    else:
        X = dataframe[x_label].values
    if y_label is None:
        y = np.zeros(dataframe.shape[0]).reshape(-1, 1)
    else:
        y = dataframe[y_label].values.reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    data = np.hstack((X, y))
    return (data, X, y)
xLabel = list(train_df_new.columns)
removea = ['PassengerId', 'Transported', 'PassengerIdGroup', 'PassengerIdNum']
for i in removea:
    xLabel.remove(i)
print(xLabel)
(_, X_, y_) = get_xy(train_df_new, 'Transported', x_label=xLabel)
(_, X_test_sub, y_test_sub) = get_xy(test_df_new, None, x_label=xLabel)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X_, y_, random_state=0, train_size=0.75)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()