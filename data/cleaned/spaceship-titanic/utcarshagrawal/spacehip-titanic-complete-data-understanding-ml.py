
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
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head()
df_test.head()
print(colored(f'Number of rows in train data: {df_train.shape[0]}', 'red'))
print(colored(f'Number of columns in train data: {df_train.shape[1]}', 'red'))
print(colored(f'Number of rows in test data: {df_test.shape[0]}', 'blue'))
print(colored(f'Number of columns in test data: {df_test.shape[1]}', 'blue'))
df_train.describe()
df_test.describe()
plt.figure(figsize=(20, 6))
na = pd.DataFrame(df_train.isna().sum())
sns.barplot(y=na[0], x=na.index)
plt.title('Missing Values Distribution', size=20, weight='bold')
print(colored('Missing values column wise -', 'magenta'))
print(colored(df_train.isna().sum(), 'magenta'))

(fig, ax) = plt.subplots(4, 3, figsize=(20, 20))
fig.suptitle('Missing Values Distribution By Transported', size=20, weight='bold')
fig.subplots_adjust(top=0.95)
i = 0
for x in df_train.columns:
    if len(df_train[df_train[x].isna() == True]) > 0:
        sns.countplot(x='Transported', data=df_train[df_train[x].isna() == True], ax=fig.axes[i], palette='turbo')
        fig.axes[i].set_title(x, weight='bold')
        i += 1

plt.figure(figsize=(9, 6))
sns.countplot(x='Transported', data=df_train, palette='winter')
plt.title('Transported Distribution', size=20, weight='bold')
print(colored(f"Percentage of Passengers Transported - {len(df_train[df_train['Transported'] == True]) / df_train.shape[0] * 100:.2f}%", 'cyan'))
print(colored(f"Percentage of Passengers Not Transported - {len(df_train[df_train['Transported'] == False]) / df_train.shape[0] * 100:.2f}%", 'cyan'))

(fig, ax) = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('GroupSize Distribution', size=20, weight='bold')
df_train['Group'] = df_train['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
df_train['GroupSize'] = df_train['Group'].map(lambda x: df_train['Group'].value_counts()[x])
df_temp = df_train.drop_duplicates(subset=['Group'], keep='last')
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
sizes = list(df_train['HomePlanet'].value_counts(sort=False))
labels = df_train['HomePlanet'].dropna().unique()
colors = ['#099FFF', '#CC00FF', '#13CA91']
explode = (0.05, 0.05, 0.05)
ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels, autopct='%1.2f%%', pctdistance=0.6, textprops={'fontsize': 12})
sns.countplot(x='HomePlanet', data=df_train, hue='Transported', ax=ax[1])
print(colored('HomePlanet Distribution - ', 'green'))
data = pd.DataFrame(df_train['HomePlanet'].value_counts()).reset_index().rename(columns={'index': 'HomePlanet', 'HomePlanet': 'Count'})
print(colored(data, 'green'))

(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('CryoSleep Distribution', size=20, weight='bold')
sizes = list(df_train['CryoSleep'].value_counts())
labels = df_train['CryoSleep'].dropna().unique()
colors = ['#099FFF', '#CC00FF']
explode = (0.05, 0.05)
ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels, autopct='%1.2f%%', pctdistance=0.6, textprops={'fontsize': 12})
sns.countplot(x='CryoSleep', data=df_train, hue='Transported', ax=ax[1])
print(colored('CryoSleep Distribution - ', 'magenta'))
data = pd.DataFrame(df_train['CryoSleep'].value_counts()).reset_index().rename(columns={'index': 'CryoSleep', 'CryoSleep': 'Count'})
print(colored(data, 'magenta'))

df_train['Cabin'].fillna('Z/9999/Z', inplace=True)
df_train['deck'] = df_train['Cabin'].apply(lambda x: x.split('/')[0])
df_train['side'] = df_train['Cabin'].apply(lambda x: x.split('/')[2])
(fig, ax) = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Cabin Distribution', size=20, weight='bold')
sns.countplot(x='deck', data=df_train, order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'Z'], ax=ax[0][0], palette='turbo')
sns.countplot(x='deck', data=df_train, order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'Z'], hue='Transported', ax=ax[0][1])
sns.countplot(x='side', data=df_train, ax=ax[1][0], palette='turbo')
sns.countplot(x='side', data=df_train, hue='Transported', ax=ax[1][1])
print(colored('Cabin Deck Distribution - ', 'red'))
data = pd.DataFrame(df_train['deck'].value_counts()).reset_index().rename(columns={'index': 'Deck', 'deck': 'Count'})
print(colored(data, 'red'))
print(colored('Cabin Side Distribution - ', 'blue'))
data = pd.DataFrame(df_train['side'].value_counts()).reset_index().rename(columns={'index': 'Side', 'side': 'Count'})
print(colored(data, 'blue'))

(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('Destination Distribution', size=20, weight='bold')
sizes = list(df_train['Destination'].value_counts(sort=False))
labels = df_train['Destination'].dropna().unique()
colors = ['#099FFF', '#CC00FF', '#13CA91']
explode = (0.05, 0.05, 0.05)
ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels, autopct='%1.2f%%', pctdistance=0.6, textprops={'fontsize': 12})
sns.countplot(x='Destination', data=df_train, hue='Transported', ax=ax[1])
print(colored('Destination Distribution - ', 'cyan'))
data = pd.DataFrame(df_train['Destination'].value_counts()).reset_index().rename(columns={'index': 'Destination', 'Destination': 'Count'})
print(colored(data, 'cyan'))

(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('VIP Distribution', size=20, weight='bold')
sizes = list(df_train['VIP'].value_counts(sort=False))
labels = df_train['VIP'].dropna().unique()
colors = ['#099FFF', '#CC00FF']
explode = (0.25, 0.25)
ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels, autopct='%1.2f%%', pctdistance=0.6, textprops={'fontsize': 12})
sns.countplot(x='VIP', data=df_train, hue='Transported', ax=ax[1])
print(colored('VIP Distribution - ', 'green'))
data = pd.DataFrame(df_train['VIP'].value_counts()).reset_index().rename(columns={'index': 'VIP', 'VIP': 'Count'})
print(colored(data, 'green'))

(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('Age Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='Age', data=df_train, ax=ax[0])
sns.histplot(x='Age', element='step', kde=True, data=df_train, hue='Transported', ax=ax[1])
print(colored('Transported Passengers Age Distribution - ', 'magenta'))
print(colored(f"Minimum Age - {df_train[df_train['Transported'] == True]['Age'].describe()['min']}", 'magenta'))
print(colored(f"Maximum Age - {df_train[df_train['Transported'] == True]['Age'].describe()['max']}", 'magenta'))
print(colored(f"Average Age - {df_train[df_train['Transported'] == True]['Age'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers Age Distribution - ', 'blue'))
print(colored(f"Minimum Age - {df_train[df_train['Transported'] == False]['Age'].describe()['min']}", 'blue'))
print(colored(f"Maximum Age - {df_train[df_train['Transported'] == False]['Age'].describe()['max']}", 'blue'))
print(colored(f"Average Age - {df_train[df_train['Transported'] == False]['Age'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()

(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('RoomService Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='RoomService', data=df_train, ax=ax[0])
sns.histplot(x='RoomService', element='step', kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1])
print(colored(f"Percentage of Passengers with no RoomService Billing - {len(df_train[df_train['RoomService'] == 0.0]) / len(df_train) * 100:.2f}%", 'red'))
print(colored('Transported Passengers RoomService Billing Distribution - ', 'magenta'))
print(colored(f"Minimum RoomService Billing - {df_train[df_train['Transported'] == True]['RoomService'].describe()['min']}", 'magenta'))
print(colored(f"Maximum RoomService Billing - {df_train[df_train['Transported'] == True]['RoomService'].describe()['max']}", 'magenta'))
print(colored(f"Average RoomService Billing - {df_train[df_train['Transported'] == True]['RoomService'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers RoomService Billing Distribution - ', 'blue'))
print(colored(f"Minimum RoomService Billing - {df_train[df_train['Transported'] == False]['RoomService'].describe()['min']}", 'blue'))
print(colored(f"Maximum RoomService Billing - {df_train[df_train['Transported'] == False]['RoomService'].describe()['max']}", 'blue'))
print(colored(f"Average RoomService Billing - {df_train[df_train['Transported'] == False]['RoomService'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()

(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('FoodCourt Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='FoodCourt', data=df_train, ax=ax[0])
sns.histplot(x='FoodCourt', element='step', kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1])
print(colored(f"Percentage of Passengers with no FoodCourt Billing - {len(df_train[df_train['FoodCourt'] == 0.0]) / len(df_train) * 100:.2f}%", 'red'))
print(colored('Transported Passengers FoodCourt Billing Distribution - ', 'magenta'))
print(colored(f"Minimum FoodCourt Billing - {df_train[df_train['Transported'] == True]['FoodCourt'].describe()['min']}", 'magenta'))
print(colored(f"Maximum FoodCourt Billing - {df_train[df_train['Transported'] == True]['FoodCourt'].describe()['max']}", 'magenta'))
print(colored(f"Average FoodCourt Billing - {df_train[df_train['Transported'] == True]['FoodCourt'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers FoodCourt Billing Distribution - ', 'blue'))
print(colored(f"Minimum FoodCourt Billing - {df_train[df_train['Transported'] == False]['FoodCourt'].describe()['min']}", 'blue'))
print(colored(f"Maximum FoodCourt Billing - {df_train[df_train['Transported'] == False]['FoodCourt'].describe()['max']}", 'blue'))
print(colored(f"Average FoodCourt Billing - {df_train[df_train['Transported'] == False]['FoodCourt'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()

(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('ShoppingMall Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='ShoppingMall', data=df_train, ax=ax[0])
sns.histplot(x='ShoppingMall', element='step', kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1])
print(colored(f"Percentage of Passengers with no ShoppingMall Billing - {len(df_train[df_train['ShoppingMall'] == 0.0]) / len(df_train) * 100:.2f}%", 'red'))
print(colored('Transported Passengers ShoppingMall Billing Distribution - ', 'magenta'))
print(colored(f"Minimum ShoppingMall Billing - {df_train[df_train['Transported'] == True]['ShoppingMall'].describe()['min']}", 'magenta'))
print(colored(f"Maximum ShoppingMall Billing - {df_train[df_train['Transported'] == True]['ShoppingMall'].describe()['max']}", 'magenta'))
print(colored(f"Average ShoppingMall Billing - {df_train[df_train['Transported'] == True]['ShoppingMall'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers ShoppingMall Billing Distribution - ', 'blue'))
print(colored(f"Minimum ShoppingMall Billing - {df_train[df_train['Transported'] == False]['ShoppingMall'].describe()['min']}", 'blue'))
print(colored(f"Maximum ShoppingMall Billing - {df_train[df_train['Transported'] == False]['ShoppingMall'].describe()['max']}", 'blue'))
print(colored(f"Average ShoppingMall Billing - {df_train[df_train['Transported'] == False]['ShoppingMall'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()

(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('Spa Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='Spa', data=df_train, ax=ax[0])
sns.histplot(x='Spa', element='step', kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1])
print(colored(f"Percentage of Passengers with no Spa Billing - {len(df_train[df_train['Spa'] == 0.0]) / len(df_train) * 100:.2f}%", 'red'))
print(colored('Transported Passengers Spa Billing Distribution - ', 'magenta'))
print(colored(f"Minimum Spa Billing - {df_train[df_train['Transported'] == True]['Spa'].describe()['min']}", 'magenta'))
print(colored(f"Maximum Spa Billing - {df_train[df_train['Transported'] == True]['Spa'].describe()['max']}", 'magenta'))
print(colored(f"Average Spa Billing - {df_train[df_train['Transported'] == True]['Spa'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers Spa Billing Distribution - ', 'blue'))
print(colored(f"Minimum Spa Billing - {df_train[df_train['Transported'] == False]['Spa'].describe()['min']}", 'blue'))
print(colored(f"Maximum Spa Billing - {df_train[df_train['Transported'] == False]['Spa'].describe()['max']}", 'blue'))
print(colored(f"Average Spa Billing - {df_train[df_train['Transported'] == False]['Spa'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()

(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('VRDeck Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='VRDeck', data=df_train, ax=ax[0])
sns.histplot(x='VRDeck', element='step', kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1])
print(colored(f"Percentage of Passengers with no VRDeck Billing - {len(df_train[df_train['VRDeck'] == 0.0]) / len(df_train) * 100:.2f}%", 'red'))
print(colored('Transported Passengers VRDeck Billing Distribution - ', 'magenta'))
print(colored(f"Minimum VRDeck Billing - {df_train[df_train['Transported'] == True]['VRDeck'].describe()['min']}", 'magenta'))
print(colored(f"Maximum VRDeck Billing - {df_train[df_train['Transported'] == True]['VRDeck'].describe()['max']}", 'magenta'))
print(colored(f"Average VRDeck Billing - {df_train[df_train['Transported'] == True]['VRDeck'].describe()['mean']}", 'magenta'))
print(colored('Non Transported Passengers VRDeck Billing Distribution - ', 'blue'))
print(colored(f"Minimum VRDeck Billing - {df_train[df_train['Transported'] == False]['VRDeck'].describe()['min']}", 'blue'))
print(colored(f"Maximum VRDeck Billing - {df_train[df_train['Transported'] == False]['VRDeck'].describe()['max']}", 'blue'))
print(colored(f"Average VRDeck Billing - {df_train[df_train['Transported'] == False]['VRDeck'].describe()['mean']}", 'blue'))
mplcyberpunk.make_lines_glow()

(fig, ax) = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('Continuous Features VS Age', size=20, weight='bold')
fig.delaxes(ax[2][1])
df_temp = df_train.iloc[:, 5:12]
columns = df_temp.columns[2:]
for (i, col) in enumerate(columns):
    sns.scatterplot(x='Age', y=col, hue='Transported', data=df_train, ax=fig.axes[i], palette='turbo')
    fig.axes[i].set_title(f'{col} VS Age', weight='bold')

(fig, ax) = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('Continuous Features VS VIP', size=20, weight='bold')
fig.delaxes(ax[2][1])
for (i, col) in enumerate(columns):
    sns.stripplot(x='VIP', y=col, hue='Transported', data=df_train, dodge=True, ax=fig.axes[i], palette='winter')
    fig.axes[i].set_title(f'{col} VS VIP', weight='bold')

df_train['Total_Expenses'] = df_train[df_temp.columns[2:]].sum(axis=1)
df_train['NoSpent'] = df_train['Total_Expenses'] == 0
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('Total Expenses Distribution', size=20, weight='bold')
sns.boxplot(x='Transported', y='Total_Expenses', data=df_train, ax=ax[0], palette='turbo')
sns.histplot(x='Total_Expenses', element='step', kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1], palette='turbo')
print(colored('Total_Expenses Distribution - ', 'cyan'))
print(colored(f"Minimum Total_Expenses - {df_train['Total_Expenses'].describe()['min']}", 'cyan'))
print(colored(f"Maximum Total_Expenses - {df_train['Total_Expenses'].describe()['max']}", 'cyan'))
print(colored(f"Average Total_Expenses - {df_train['Total_Expenses'].describe()['mean']}", 'cyan'))

(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('NoSpent Distribution', size=20, weight='bold')
sizes = list(df_train['NoSpent'].value_counts(sort=False))
labels = df_train['NoSpent'].dropna().unique()
colors = ['#13CA91', '#e5ab09']
explode = (0.0, 0.05)
ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels, autopct='%1.2f%%', pctdistance=0.6, textprops={'fontsize': 12})
sns.countplot(x='NoSpent', data=df_train, hue='Transported', ax=ax[1], palette='turbo')
print(colored('NoSpent Distribution - ', 'cyan'))
data = pd.DataFrame(df_train['NoSpent'].value_counts()).reset_index().rename(columns={'index': 'NoSpent', 'NoSpent': 'Count'})
print(colored(data, 'cyan'))

df_train.head()
print(colored(f'Number of missing values before - {df_train.isna().sum().sum()}', 'red'))
for col in df_train.columns:
    if col == 'Age':
        df_train[col].fillna(df_train[col].median(), inplace=True)
    else:
        df_train[col].fillna(df_train[col].mode()[0], inplace=True)
print(colored(f'Number of missing values after - {df_train.isna().sum().sum()}', 'blue'))
df_train.drop(['PassengerId', 'Cabin', 'Group', 'Name'], axis=1, inplace=True)
for col in df_train.columns[df_train.dtypes == object]:
    encoder = LabelEncoder()
    df_train[col] = encoder.fit_transform(df_train[col])
for col in df_train.columns[df_train.dtypes == bool]:
    df_train[col] = df_train[col].astype('int')
X = df_train.drop('Transported', axis=1)
y = df_train['Transported']
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, stratify=y, test_size=0.2, random_state=5)
print(colored(f'Number of rows in training set - {len(X_train)}', 'cyan'))
print(colored(f'Number of rows in validation set - {len(X_valid)}', 'magenta'))
acc_plot = {}
preds = []