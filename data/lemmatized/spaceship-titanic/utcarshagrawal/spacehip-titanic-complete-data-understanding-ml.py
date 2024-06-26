import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplcyberpunk
from termcolor import colored
plt.style.use('cyberpunk')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
print(colored(f'Number of rows in train data: {_input1.shape[0]}', 'red'))
print(colored(f'Number of columns in train data: {_input1.shape[1]}', 'red'))
print(colored(f'Number of rows in test data: {_input0.shape[0]}', 'blue'))
print(colored(f'Number of columns in test data: {_input0.shape[1]}', 'blue'))
_input1.describe()
_input0.describe()
plt.figure(figsize=(20, 6))
na = pd.DataFrame(_input1.isna().sum())
sns.barplot(y=na[0], x=na.index)
plt.title('Missing Values Distribution', size=20, weight='bold')
print(colored('Missing values column wise -', 'magenta'))
print(colored(_input1.isna().sum(), 'magenta'))
(fig, ax) = plt.subplots(4, 3, figsize=(20, 20))
fig.suptitle('Missing Values Distribution By Transported', size=20, weight='bold')
fig.subplots_adjust(top=0.95)
i = 0
for x in _input1.columns:
    if len(_input1[_input1[x].isna() == True]) > 0:
        sns.countplot(x='Transported', data=_input1[_input1[x].isna() == True], ax=fig.axes[i], palette='turbo')
        fig.axes[i].set_title(x, weight='bold')
        i += 1
plt.figure(figsize=(9, 6))
sns.countplot(x='Transported', data=_input1, palette='winter')
plt.title('Transported Distribution', size=20, weight='bold')
print(colored(f"Percentage of Passengers Transported - {len(_input1[_input1['Transported'] == True]) / _input1.shape[0] * 100:.2f}%", 'cyan'))
print(colored(f"Percentage of Passengers Not Transported - {len(_input1[_input1['Transported'] == False]) / _input1.shape[0] * 100:.2f}%", 'cyan'))
(fig, ax) = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('GroupSize Distribution', size=20, weight='bold')
_input1['Group'] = _input1['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
_input1['GroupSize'] = _input1['Group'].map(lambda x: _input1['Group'].value_counts()[x])
df_temp = _input1.drop_duplicates(subset=['Group'], keep='last')
sns.countplot(x='GroupSize', data=df_temp, ax=ax[0])
sns.countplot(x='GroupSize', data=df_temp, hue='Transported', ax=ax[1])
print(colored(f'Number of unique groups - {len(df_temp)}', 'blue'))
data = pd.DataFrame(df_temp['GroupSize'].value_counts()).reset_index().rename(columns={'index': 'GroupSize', 'GroupSize': 'Count'})
print(colored('Group Size Distribution - ', 'blue'))
print(colored(data, 'blue'))
print(colored(f"Total number of individual passengers - {len(df_temp[df_temp['GroupSize'] == 1])}", 'blue'))
print(colored(f"Number of individual passengers transported - {len(df_temp[(df_temp['GroupSize'] == 1) & (df_temp['Transported'] == True)])}", 'blue'))
print(colored(f"Number of individual passengers not transported - {len(df_temp[(df_temp['GroupSize'] == 1) & (df_temp['Transported'] == False)])}", 'blue'))
print(colored(f"Toal number of non individual passengers - {len(df_temp[df_temp['GroupSize'] != 1])}", 'red'))
print(colored(f"Number of non individual passengers transported - {len(df_temp[(df_temp['GroupSize'] != 1) & (df_temp['Transported'] == True)])}", 'red'))
print(colored(f"Number of non individual passengers not transported - {len(df_temp[(df_temp['GroupSize'] != 1) & (df_temp['Transported'] == False)])}", 'red'))
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('HomePlanet Distribution', size=20, weight='bold')
sizes = list(_input1['HomePlanet'].value_counts(sort=False))
labels = _input1['HomePlanet'].dropna().unique()
colors = ['#099FFF', '#CC00FF', '#13CA91']
explode = (0.05, 0.05, 0.05)
ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels, autopct='%1.2f%%', pctdistance=0.6, textprops={'fontsize': 12})
sns.countplot(x='HomePlanet', data=_input1, hue='Transported', ax=ax[1])
print(colored('HomePlanet Distribution - ', 'green'))
data = pd.DataFrame(_input1['HomePlanet'].value_counts()).reset_index().rename(columns={'index': 'HomePlanet', 'HomePlanet': 'Count'})
print(colored(data, 'green'))
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('CryoSleep Distribution', size=20, weight='bold')
sizes = list(_input1['CryoSleep'].value_counts())
labels = _input1['CryoSleep'].dropna().unique()
colors = ['#099FFF', '#CC00FF']
explode = (0.05, 0.05)
ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels, autopct='%1.2f%%', pctdistance=0.6, textprops={'fontsize': 12})
sns.countplot(x='CryoSleep', data=_input1, hue='Transported', ax=ax[1])
print(colored('CryoSleep Distribution - ', 'magenta'))
data = pd.DataFrame(_input1['CryoSleep'].value_counts()).reset_index().rename(columns={'index': 'CryoSleep', 'CryoSleep': 'Count'})
print(colored(data, 'magenta'))
_input1['Cabin'] = _input1['Cabin'].fillna('Z/9999/Z', inplace=False)
_input1['deck'] = _input1['Cabin'].apply(lambda x: x.split('/')[0])
_input1['side'] = _input1['Cabin'].apply(lambda x: x.split('/')[2])
(fig, ax) = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Cabin Distribution', size=20, weight='bold')
sns.countplot(x='deck', data=_input1, order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'Z'], ax=ax[0][0], palette='turbo')
sns.countplot(x='deck', data=_input1, order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'Z'], hue='Transported', ax=ax[0][1])
sns.countplot(x='side', data=_input1, ax=ax[1][0], palette='turbo')
sns.countplot(x='side', data=_input1, hue='Transported', ax=ax[1][1])
print(colored('Cabin Deck Distribution - ', 'red'))
data = pd.DataFrame(_input1['deck'].value_counts()).reset_index().rename(columns={'index': 'Deck', 'deck': 'Count'})
print(colored(data, 'red'))
print(colored('Cabin Side Distribution - ', 'blue'))
data = pd.DataFrame(_input1['side'].value_counts()).reset_index().rename(columns={'index': 'Side', 'side': 'Count'})
print(colored(data, 'blue'))
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('Destination Distribution', size=20, weight='bold')
sizes = list(_input1['Destination'].value_counts(sort=False))
labels = _input1['Destination'].dropna().unique()
colors = ['#099FFF', '#CC00FF', '#13CA91']
explode = (0.05, 0.05, 0.05)
ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels, autopct='%1.2f%%', pctdistance=0.6, textprops={'fontsize': 12})
sns.countplot(x='Destination', data=_input1, hue='Transported', ax=ax[1])
print(colored('Destination Distribution - ', 'cyan'))
data = pd.DataFrame(_input1['Destination'].value_counts()).reset_index().rename(columns={'index': 'Destination', 'Destination': 'Count'})
print(colored(data, 'cyan'))
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('VIP Distribution', size=20, weight='bold')
sizes = list(_input1['VIP'].value_counts(sort=False))
labels = _input1['VIP'].dropna().unique()
colors = ['#099FFF', '#CC00FF']
explode = (0.25, 0.25)
ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels, autopct='%1.2f%%', pctdistance=0.6, textprops={'fontsize': 12})
sns.countplot(x='VIP', data=_input1, hue='Transported', ax=ax[1])
print(colored('VIP Distribution - ', 'green'))
data = pd.DataFrame(_input1['VIP'].value_counts()).reset_index().rename(columns={'index': 'VIP', 'VIP': 'Count'})
print(colored(data, 'green'))
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('Age Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='Age', data=_input1, ax=ax[0])
sns.histplot(x='Age', element='step', kde=True, data=_input1, hue='Transported', ax=ax[1])
print(colored('Transported Passengers Age Distribution - ', 'magenta'))
print(colored(f"Minimum Age - {_input1[_input1['Transported'] == True]['Age'].describe()['min']}", 'magenta'))
print(colored(f"Maximum Age - {_input1[_input1['Transported'] == True]['Age'].describe()['max']}", 'magenta'))
print(colored(f"Average Age - {_input1[_input1['Transported'] == True]['Age'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers Age Distribution - ', 'blue'))
print(colored(f"Minimum Age - {_input1[_input1['Transported'] == False]['Age'].describe()['min']}", 'blue'))
print(colored(f"Maximum Age - {_input1[_input1['Transported'] == False]['Age'].describe()['max']}", 'blue'))
print(colored(f"Average Age - {_input1[_input1['Transported'] == False]['Age'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('RoomService Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='RoomService', data=_input1, ax=ax[0])
sns.histplot(x='RoomService', element='step', kde=True, data=_input1, hue='Transported', bins=100, ax=ax[1])
print(colored(f"Percentage of Passengers with no RoomService Billing - {len(_input1[_input1['RoomService'] == 0.0]) / len(_input1) * 100:.2f}%", 'red'))
print(colored('Transported Passengers RoomService Billing Distribution - ', 'magenta'))
print(colored(f"Minimum RoomService Billing - {_input1[_input1['Transported'] == True]['RoomService'].describe()['min']}", 'magenta'))
print(colored(f"Maximum RoomService Billing - {_input1[_input1['Transported'] == True]['RoomService'].describe()['max']}", 'magenta'))
print(colored(f"Average RoomService Billing - {_input1[_input1['Transported'] == True]['RoomService'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers RoomService Billing Distribution - ', 'blue'))
print(colored(f"Minimum RoomService Billing - {_input1[_input1['Transported'] == False]['RoomService'].describe()['min']}", 'blue'))
print(colored(f"Maximum RoomService Billing - {_input1[_input1['Transported'] == False]['RoomService'].describe()['max']}", 'blue'))
print(colored(f"Average RoomService Billing - {_input1[_input1['Transported'] == False]['RoomService'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('FoodCourt Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='FoodCourt', data=_input1, ax=ax[0])
sns.histplot(x='FoodCourt', element='step', kde=True, data=_input1, hue='Transported', bins=100, ax=ax[1])
print(colored(f"Percentage of Passengers with no FoodCourt Billing - {len(_input1[_input1['FoodCourt'] == 0.0]) / len(_input1) * 100:.2f}%", 'red'))
print(colored('Transported Passengers FoodCourt Billing Distribution - ', 'magenta'))
print(colored(f"Minimum FoodCourt Billing - {_input1[_input1['Transported'] == True]['FoodCourt'].describe()['min']}", 'magenta'))
print(colored(f"Maximum FoodCourt Billing - {_input1[_input1['Transported'] == True]['FoodCourt'].describe()['max']}", 'magenta'))
print(colored(f"Average FoodCourt Billing - {_input1[_input1['Transported'] == True]['FoodCourt'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers FoodCourt Billing Distribution - ', 'blue'))
print(colored(f"Minimum FoodCourt Billing - {_input1[_input1['Transported'] == False]['FoodCourt'].describe()['min']}", 'blue'))
print(colored(f"Maximum FoodCourt Billing - {_input1[_input1['Transported'] == False]['FoodCourt'].describe()['max']}", 'blue'))
print(colored(f"Average FoodCourt Billing - {_input1[_input1['Transported'] == False]['FoodCourt'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('ShoppingMall Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='ShoppingMall', data=_input1, ax=ax[0])
sns.histplot(x='ShoppingMall', element='step', kde=True, data=_input1, hue='Transported', bins=100, ax=ax[1])
print(colored(f"Percentage of Passengers with no ShoppingMall Billing - {len(_input1[_input1['ShoppingMall'] == 0.0]) / len(_input1) * 100:.2f}%", 'red'))
print(colored('Transported Passengers ShoppingMall Billing Distribution - ', 'magenta'))
print(colored(f"Minimum ShoppingMall Billing - {_input1[_input1['Transported'] == True]['ShoppingMall'].describe()['min']}", 'magenta'))
print(colored(f"Maximum ShoppingMall Billing - {_input1[_input1['Transported'] == True]['ShoppingMall'].describe()['max']}", 'magenta'))
print(colored(f"Average ShoppingMall Billing - {_input1[_input1['Transported'] == True]['ShoppingMall'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers ShoppingMall Billing Distribution - ', 'blue'))
print(colored(f"Minimum ShoppingMall Billing - {_input1[_input1['Transported'] == False]['ShoppingMall'].describe()['min']}", 'blue'))
print(colored(f"Maximum ShoppingMall Billing - {_input1[_input1['Transported'] == False]['ShoppingMall'].describe()['max']}", 'blue'))
print(colored(f"Average ShoppingMall Billing - {_input1[_input1['Transported'] == False]['ShoppingMall'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('Spa Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='Spa', data=_input1, ax=ax[0])
sns.histplot(x='Spa', element='step', kde=True, data=_input1, hue='Transported', bins=100, ax=ax[1])
print(colored(f"Percentage of Passengers with no Spa Billing - {len(_input1[_input1['Spa'] == 0.0]) / len(_input1) * 100:.2f}%", 'red'))
print(colored('Transported Passengers Spa Billing Distribution - ', 'magenta'))
print(colored(f"Minimum Spa Billing - {_input1[_input1['Transported'] == True]['Spa'].describe()['min']}", 'magenta'))
print(colored(f"Maximum Spa Billing - {_input1[_input1['Transported'] == True]['Spa'].describe()['max']}", 'magenta'))
print(colored(f"Average Spa Billing - {_input1[_input1['Transported'] == True]['Spa'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers Spa Billing Distribution - ', 'blue'))
print(colored(f"Minimum Spa Billing - {_input1[_input1['Transported'] == False]['Spa'].describe()['min']}", 'blue'))
print(colored(f"Maximum Spa Billing - {_input1[_input1['Transported'] == False]['Spa'].describe()['max']}", 'blue'))
print(colored(f"Average Spa Billing - {_input1[_input1['Transported'] == False]['Spa'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('VRDeck Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='VRDeck', data=_input1, ax=ax[0])
sns.histplot(x='VRDeck', element='step', kde=True, data=_input1, hue='Transported', bins=100, ax=ax[1])
print(colored(f"Percentage of Passengers with no VRDeck Billing - {len(_input1[_input1['VRDeck'] == 0.0]) / len(_input1) * 100:.2f}%", 'red'))
print(colored('Transported Passengers VRDeck Billing Distribution - ', 'magenta'))
print(colored(f"Minimum VRDeck Billing - {_input1[_input1['Transported'] == True]['VRDeck'].describe()['min']}", 'magenta'))
print(colored(f"Maximum VRDeck Billing - {_input1[_input1['Transported'] == True]['VRDeck'].describe()['max']}", 'magenta'))
print(colored(f"Average VRDeck Billing - {_input1[_input1['Transported'] == True]['VRDeck'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers VRDeck Billing Distribution - ', 'blue'))
print(colored(f"Minimum VRDeck Billing - {_input1[_input1['Transported'] == False]['VRDeck'].describe()['min']}", 'blue'))
print(colored(f"Maximum VRDeck Billing - {_input1[_input1['Transported'] == False]['VRDeck'].describe()['max']}", 'blue'))
print(colored(f"Average VRDeck Billing - {_input1[_input1['Transported'] == False]['VRDeck'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()
(fig, ax) = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('Continuous Features VS Age', size=20, weight='bold')
fig.delaxes(ax[2][1])
df_temp = _input1.iloc[:, 5:12]
columns = df_temp.columns[2:]
for (i, col) in enumerate(columns):
    sns.scatterplot(x='Age', y=col, hue='Transported', data=_input1, ax=fig.axes[i], palette='turbo')
    fig.axes[i].set_title(f'{col} VS Age', weight='bold')
(fig, ax) = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('Continuous Features VS VIP', size=20, weight='bold')
fig.delaxes(ax[2][1])
for (i, col) in enumerate(columns):
    sns.stripplot(x='VIP', y=col, hue='Transported', data=_input1, dodge=True, ax=fig.axes[i], palette='winter')
    fig.axes[i].set_title(f'{col} VS VIP', weight='bold')
_input1['Total_Expenses'] = _input1[df_temp.columns[2:]].sum(axis=1)
_input1['NoSpent'] = _input1['Total_Expenses'] == 0
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('Total Expenses Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='Total_Expenses', data=_input1, ax=ax[0], palette='turbo')
sns.histplot(x='Total_Expenses', element='step', kde=True, data=_input1, hue='Transported', bins=100, ax=ax[1], palette='turbo')
print(colored('Total_Expenses Distribution - ', 'cyan'))
print(colored(f"Minimum Total_Expenses - {_input1['Total_Expenses'].describe()['min']}", 'cyan'))
print(colored(f"Maximum Total_Expenses - {_input1['Total_Expenses'].describe()['max']}", 'cyan'))
print(colored(f"Average Total_Expenses - {_input1['Total_Expenses'].describe()['mean']}", 'cyan'))
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('NoSpent Distribution', size=20, weight='bold')
sizes = list(_input1['NoSpent'].value_counts(sort=False))
labels = _input1['NoSpent'].dropna().unique()
colors = ['#13CA91', '#e5ab09']
explode = (0.0, 0.05)
ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels, autopct='%1.2f%%', pctdistance=0.6, textprops={'fontsize': 12})
sns.countplot(x='NoSpent', data=_input1, hue='Transported', ax=ax[1], palette='turbo')
print(colored('NoSpent Distribution - ', 'cyan'))
data = pd.DataFrame(_input1['NoSpent'].value_counts()).reset_index().rename(columns={'index': 'NoSpent', 'NoSpent': 'Count'})
print(colored(data, 'cyan'))
_input1.head()
print(colored(f'Number of missing values before - {_input1.isna().sum().sum()}', 'red'))
for col in _input1.columns:
    if col == 'Age':
        _input1[col] = _input1[col].fillna(_input1[col].median(), inplace=False)
    else:
        _input1[col] = _input1[col].fillna(_input1[col].mode()[0], inplace=False)
print(colored(f'Number of missing values after - {_input1.isna().sum().sum()}', 'blue'))
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Group', 'Name'], axis=1, inplace=False)
for col in _input1.columns[_input1.dtypes == object]:
    encoder = LabelEncoder()
    _input1[col] = encoder.fit_transform(_input1[col])
for col in _input1.columns[_input1.dtypes == bool]:
    _input1[col] = _input1[col].astype('int')
X = _input1.drop('Transported', axis=1)
y = _input1['Transported']
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, stratify=y, test_size=0.2, random_state=5)
print(colored(f'Number of rows in training set - {len(X_train)}', 'cyan'))
print(colored(f'Number of rows in validation set - {len(X_valid)}', 'magenta'))
acc_plot = {}
preds = []