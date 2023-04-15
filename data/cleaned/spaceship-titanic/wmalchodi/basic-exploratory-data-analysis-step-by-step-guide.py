import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import warnings
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.factorplots import interaction_plot
from scipy.stats import iqr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
sns.set_style('darkgrid')
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head()
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test.head()
train.dtypes
train['Transported'].isnull().sum()
train['Transported'].value_counts(normalize=True)
train.isna().sum()
test.isna().sum()
train['HomePlanet'].value_counts(normalize=True)
train['HomePlanet'] = np.where(train['HomePlanet'].isnull(), 'Unknown', train['HomePlanet'])
test['HomePlanet'] = np.where(test['HomePlanet'].isnull(), 'Unknown', test['HomePlanet'])
train['HomePlanet'].value_counts(normalize=True)
test['HomePlanet'].value_counts(normalize=True)
(fig, ax) = plt.subplots(figsize=(6, 6))
sns.histplot(data=train, multiple='fill', x='HomePlanet', kde=False, stat='proportion', palette='pastel', hue='Transported', element='bars', legend=True, discrete=True)
plt.title('Proportion of Transported Passengers for Each HomePlanet', fontsize=15)
plt.ylabel('Proportion', fontsize=10)
plt.xlabel('Home Planet', fontsize=10)

train['CryoSleep'].value_counts(normalize=True)
train['CryoSleep'] = np.where(train['CryoSleep'] == True, 'yes', np.where(train['CryoSleep'] == False, 'no', 'unknown'))
test['CryoSleep'] = np.where(test['CryoSleep'] == True, 'yes', np.where(test['CryoSleep'] == False, 'no', 'unknown'))
train['CryoSleep'].value_counts(normalize=True)
test['CryoSleep'].value_counts(normalize=True)
(fig, ax) = plt.subplots(figsize=(6, 6))
sns.histplot(data=train, multiple='fill', x='CryoSleep', kde=False, stat='proportion', palette='pastel', hue='Transported', element='bars', legend=True, discrete=True)
plt.title('Proportion of Transported Passengers for Each CryoSleep', fontsize=15)
plt.ylabel('Proportion', fontsize=10)
plt.xlabel('CryoSleep', fontsize=10)

train['Cabin'] = np.where(train['Cabin'].isnull(), '-1/-1/-1', train['Cabin'])
test['Cabin'] = np.where(test['Cabin'].isnull(), '-1/-1/-1', test['Cabin'])
train['CabinDeck'] = train['Cabin'].astype(str).str[0]
test['CabinDeck'] = test['Cabin'].astype(str).str[0]
train['CabinDeck'].value_counts(normalize=True)
test['CabinDeck'].value_counts(normalize=True)
train['CabinDeck'] = np.where(train['CabinDeck'] == '-', 'Unknown', train['CabinDeck'])
test['CabinDeck'] = np.where(test['CabinDeck'] == '-', 'Unknown', test['CabinDeck'])
train['CabinDeck'].value_counts(normalize=True)
(fig, ax) = plt.subplots(figsize=(6, 6))
sns.histplot(data=train, multiple='fill', x='CabinDeck', kde=False, stat='proportion', palette='pastel', hue='Transported', element='bars', legend=True, discrete=True)
plt.title('Proportion of Transported Passengers for Each CabinDeck', fontsize=15)
plt.ylabel('Proportion', fontsize=10)
plt.xlabel('Cabin Deck', fontsize=10)

train['CabinSide'] = train['Cabin'].astype(str).str[-1]
test['CabinSide'] = test['Cabin'].astype(str).str[-1]
train['CabinSide'].value_counts(normalize=True)
test['CabinSide'].value_counts(normalize=True)
train['CabinSide'] = np.where(train['CabinSide'] == '1', 'Unknown', train['CabinSide'])
test['CabinSide'] = np.where(test['CabinSide'] == '1', 'Unknown', test['CabinSide'])
train['CabinSide'].value_counts(normalize=True)
(fig, ax) = plt.subplots(figsize=(6, 6))
sns.histplot(data=train, multiple='fill', x='CabinSide', kde=False, stat='proportion', palette='pastel', hue='Transported', element='bars', legend=True, discrete=True)
plt.title('Proportion of Transported Passengers for Each CabinSide', fontsize=15)
plt.ylabel('Proportion', fontsize=10)
plt.xlabel('Cabin Side', fontsize=10)

train['Destination'].value_counts(normalize=True)
train['Destination'] = np.where(train['Destination'].isnull(), 'Unknown', train['Destination'])
test['Destination'] = np.where(test['Destination'].isnull(), 'Unknown', test['Destination'])
train['Destination'].value_counts(normalize=True)
test['Destination'].value_counts(normalize=True)
(fig, ax) = plt.subplots(figsize=(6, 6))
sns.histplot(data=train, multiple='fill', x='Destination', kde=False, stat='proportion', palette='pastel', hue='Transported', element='bars', legend=True, discrete=True)
plt.title('Proportion of Transported Passengers for Each Destination', fontsize=15)
plt.ylabel('Proportion', fontsize=10)
plt.xlabel('Destination', fontsize=10)

train['Age'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
np.sum(train['Age'] == 0)
train['Age'] = np.where(train['Age'].isnull(), train['Age'].median(), train['Age'])
test['Age'] = np.where(test['Age'].isnull(), train['Age'].median(), test['Age'])
train['Age'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
test['Age'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
train['VIP'].value_counts()
train['VIP'] = np.where(train['VIP'] == True, 1, 0)
test['VIP'] = np.where(test['VIP'] == True, 1, 0)
train['VIP'].value_counts(normalize=True)
test['VIP'].value_counts(normalize=True)
train['RoomService'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
train['RoomService'] = np.where(train['RoomService'].isnull(), train['RoomService'].median(), train['RoomService'])
test['RoomService'] = np.where(test['RoomService'].isnull(), train['RoomService'].median(), test['RoomService'])
train['RoomService'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
test['RoomService'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
train['FoodCourt'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
train['FoodCourt'] = np.where(train['FoodCourt'].isnull(), train['FoodCourt'].median(), train['FoodCourt'])
test['FoodCourt'] = np.where(test['FoodCourt'].isnull(), train['FoodCourt'].median(), test['FoodCourt'])
train['FoodCourt'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
test['FoodCourt'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
train['ShoppingMall'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
train['ShoppingMall'] = np.where(train['ShoppingMall'].isnull(), train['ShoppingMall'].median(), train['ShoppingMall'])
test['ShoppingMall'] = np.where(test['ShoppingMall'].isnull(), train['ShoppingMall'].median(), test['ShoppingMall'])
train['ShoppingMall'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
test['ShoppingMall'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
train['Spa'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
train['Spa'] = np.where(train['Spa'].isnull(), train['Spa'].median(), train['Spa'])
test['Spa'] = np.where(test['Spa'].isnull(), train['Spa'].median(), test['Spa'])
train['Spa'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
test['Spa'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
train['VRDeck'].describe()
train['VRDeck'] = np.where(train['VRDeck'].isnull(), train['VRDeck'].median(), train['VRDeck'])
test['VRDeck'] = np.where(test['VRDeck'].isnull(), train['VRDeck'].median(), test['VRDeck'])
train['VRDeck'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
test['VRDeck'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
name_check = train[train['Name'] == 'Grake Porki']
name_check
(fig, ax) = plt.subplots()
age_plot = sns.distplot(train.Age, hist_kws=dict(edgecolor='black'))
plt.title('Age Distribution', fontsize=15)
plt.ylabel('Percentage', fontsize=10)
plt.xlabel('Age', fontsize=10)

age_iqr = iqr(train['Age'])
age_iqr_edge = age_iqr * 1.5
age_iqr_edge
(fig, axes) = plt.subplots(2, 3, figsize=(12, 12), constrained_layout=True)
room_plot = sns.distplot(train.RoomService, hist_kws=dict(edgecolor='black'), ax=axes[0, 0])
axes[0, 0].set_title('Room Service Distribution', fontsize=15)
axes[0, 0].set_ylabel('Percentage', fontsize=10)
axes[0, 0].set_xlabel('Amount', fontsize=10)
food_plot = sns.distplot(train.FoodCourt, hist_kws=dict(edgecolor='black'), ax=axes[0, 1], color='red')
axes[0, 1].set_title('Food Court Distribution', fontsize=15)
axes[0, 1].set_ylabel('Percentage', fontsize=10)
axes[0, 1].set_xlabel('Amount', fontsize=10)
shop_plot = sns.distplot(train.ShoppingMall, hist_kws=dict(edgecolor='black'), ax=axes[0, 2], color='green')
axes[0, 2].set_title('Shopping Mall Distribution', fontsize=15)
axes[0, 2].set_ylabel('Percentage', fontsize=10)
axes[0, 2].set_xlabel('Amount', fontsize=10)
spa_plot = sns.distplot(train.Spa, hist_kws=dict(edgecolor='black'), ax=axes[1, 0], color='orange')
axes[1, 0].set_title('Spa Distribution', fontsize=15)
axes[1, 0].set_ylabel('Percentage', fontsize=10)
axes[1, 0].set_xlabel('Amount', fontsize=10)
vrdeck_plot = sns.distplot(train.VRDeck, hist_kws=dict(edgecolor='black'), ax=axes[1, 1], color='purple')
axes[1, 1].set_title('VRDeck Distribution', fontsize=15)
axes[1, 1].set_ylabel('Percentage', fontsize=10)
axes[1, 1].set_xlabel('Amount', fontsize=10)
fig.delaxes(axes[1, 2])

cap_list = ['RoomService_Capped', 'FoodCourt_Capped', 'ShoppingMall_Capped', 'Spa_Capped', 'VRDeck_Capped']
var_list = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for (i, j) in zip(cap_list, var_list):
    train[i] = np.where(train[j] > train[j].quantile(0.99), train[j].quantile(0.99), train[j])
    test[i] = np.where(test[j] > test[j].quantile(0.99), test[j].quantile(0.99), test[j])
(fig, axes) = plt.subplots(2, 3, figsize=(12, 12), constrained_layout=True)
room_plot = sns.distplot(train.RoomService_Capped, hist_kws=dict(edgecolor='black'), ax=axes[0, 0])
axes[0, 0].set_title('Room Service Capped Distribution', fontsize=15)
axes[0, 0].set_ylabel('Percentage', fontsize=10)
axes[0, 0].set_xlabel('Amount', fontsize=10)
food_plot = sns.distplot(train.FoodCourt_Capped, hist_kws=dict(edgecolor='black'), ax=axes[0, 1], color='red')
axes[0, 1].set_title('Food Court Capped Distribution', fontsize=15)
axes[0, 1].set_ylabel('Percentage', fontsize=10)
axes[0, 1].set_xlabel('Amount', fontsize=10)
shop_plot = sns.distplot(train.ShoppingMall_Capped, hist_kws=dict(edgecolor='black'), ax=axes[0, 2], color='green')
axes[0, 2].set_title('Shopping Mall Capped Distribution', fontsize=15)
axes[0, 2].set_ylabel('Percentage', fontsize=10)
axes[0, 2].set_xlabel('Amount', fontsize=10)
spa_plot = sns.distplot(train.Spa_Capped, hist_kws=dict(edgecolor='black'), ax=axes[1, 0], color='orange')
axes[1, 0].set_title('Spa Capped Distribution', fontsize=15)
axes[1, 0].set_ylabel('Percentage', fontsize=10)
axes[1, 0].set_xlabel('Amount', fontsize=10)
vrdeck_plot = sns.distplot(train.VRDeck_Capped, hist_kws=dict(edgecolor='black'), ax=axes[1, 1], color='purple')
axes[1, 1].set_title('VRDeck Capped Distribution', fontsize=15)
axes[1, 1].set_ylabel('Percentage', fontsize=10)
axes[1, 1].set_xlabel('Amount', fontsize=10)
fig.delaxes(axes[1, 2])

bin_list = ['RoomService_ind', 'FoodCourt_ind', 'ShoppingMall_ind', 'Spa_ind', 'VRDeck_ind']
var_list = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for (i, j) in zip(bin_list, var_list):
    train[i] = np.where(train[j] > 0, 1, 0)
    test[i] = np.where(test[j] > 0, 1, 0)
count_list = ['RoomService_ind', 'FoodCourt_ind', 'ShoppingMall_ind', 'Spa_ind', 'VRDeck_ind']
for i in count_list:
    print(train[i].value_counts(normalize=True))
    print()
log_list = ['RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log']
var_list = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for (i, j) in zip(log_list, var_list):
    train[i] = np.log(train[j] + 1)
    test[i] = np.log(test[j] + 1)
(fig, axes) = plt.subplots(2, 3, figsize=(12, 12), constrained_layout=True)
room_plot = sns.distplot(train.RoomService_log, hist_kws=dict(edgecolor='black'), ax=axes[0, 0])
axes[0, 0].set_title('Room Service Log-transformation', fontsize=15)
axes[0, 0].set_ylabel('Percentage', fontsize=10)
axes[0, 0].set_xlabel('Amount', fontsize=10)
food_plot = sns.distplot(train.FoodCourt_log, hist_kws=dict(edgecolor='black'), ax=axes[0, 1], color='red')
axes[0, 1].set_title('Food Court Log-transformation', fontsize=15)
axes[0, 1].set_ylabel('Percentage', fontsize=10)
axes[0, 1].set_xlabel('Amount', fontsize=10)
shop_plot = sns.distplot(train.ShoppingMall_log, hist_kws=dict(edgecolor='black'), ax=axes[0, 2], color='green')
axes[0, 2].set_title('Shopping Mall Log-transformation', fontsize=15)
axes[0, 2].set_ylabel('Percentage', fontsize=10)
axes[0, 2].set_xlabel('Amount', fontsize=10)
spa_plot = sns.distplot(train.Spa_log, hist_kws=dict(edgecolor='black'), ax=axes[1, 0], color='orange')
axes[1, 0].set_title('Spa Log-transformation', fontsize=15)
axes[1, 0].set_ylabel('Percentage', fontsize=10)
axes[1, 0].set_xlabel('Amount', fontsize=10)
vrdeck_plot = sns.distplot(train.VRDeck_log, hist_kws=dict(edgecolor='black'), ax=axes[1, 1], color='purple')
axes[1, 1].set_title('VRDeck Log-transformation', fontsize=15)
axes[1, 1].set_ylabel('Percentage', fontsize=10)
axes[1, 1].set_xlabel('Amount', fontsize=10)
fig.delaxes(axes[1, 2])


def int_plot_func(x):
    (fig, axes) = plt.subplots(2, 3, figsize=(12, 12), constrained_layout=True)
    room = interaction_plot(x=train[x], trace=train['RoomService_ind'], response=train['Transported'], ax=axes[0, 0])
    axes[0, 0].set_title('Interaction of RoomService', fontsize=12)
    food = interaction_plot(x=train[x], trace=train['FoodCourt_ind'], response=train['Transported'], ax=axes[0, 1])
    axes[0, 1].set_title('Interaction of FoodCourt', fontsize=12)
    shop = interaction_plot(x=train[x], trace=train['ShoppingMall_ind'], response=train['Transported'], ax=axes[0, 2])
    axes[0, 2].set_title('Interaction of ShoppingMall', fontsize=12)
    spa = interaction_plot(x=train[x], trace=train['Spa_ind'], response=train['Transported'], ax=axes[1, 0])
    axes[1, 0].set_title('Interaction of Spa', fontsize=12)
    vr = interaction_plot(x=train[x], trace=train['VRDeck_ind'], response=train['Transported'], ax=axes[1, 1])
    axes[1, 1].set_title('Interaction of VRDeck', fontsize=12)
    fig.delaxes(axes[1, 2])
    return fig
int_plot_func('HomePlanet')

int_plot_func('CabinDeck')

int_plot_func('CabinSide')

int_plot_func('Destination')
