import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')

sns.set(style='white', context='notebook', palette='pastel')
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
data = train
print(data.info())

print(data['Transported'].value_counts(), '\n')
(fig, ax) = plt.subplots(1, 1, figsize=(5, 2))
data['Transported'].value_counts().plot(kind='pie', autopct='%.1f%%', textprops={'fontsize': 8}, startangle=90).set_title('Transported Rate')
ax.set_ylabel('')

mask_transported = data['Transported'] == True
print('Nulls:', data.PassengerId.isnull().sum(), '\n')
print(data.PassengerId.head())
data['GroupId'] = data['PassengerId'].str.split('_', expand=True)[0]
GroupId_size_serie = data.groupby('GroupId').size()
GroupId_size_serie.rename('GroupSize', inplace=True)
data = pd.merge(left=data, right=GroupId_size_serie, left_on='GroupId', right_on='GroupId', how='left')
print(data['GroupSize'].value_counts())
(fig, ax) = plt.subplots(1, 1, figsize=(5, 3))
sns.countplot(y=data['GroupSize'], palette='Set2')
ax.set_ylabel('')
ax.set_xlabel('Passenger Count')
plt.title('GroupSize')

data.loc[:, 'IsAlone'] = data['GroupSize'] == 1
print(data['IsAlone'].value_counts(), '\n')
(fig, ax) = plt.subplots(1, 2, figsize=(12, 2))
data['IsAlone'].value_counts(dropna=False).sort_index(ascending=False).plot(kind='pie', autopct='%.1f%%', ax=ax[0], cmap='Set2', startangle=90, textprops={'fontsize': 8})
ax[0].set_ylabel('')
ax[0].set_title('On-Board Rate')
sns.countplot(y=data['IsAlone'], hue=data['Transported'], ax=ax[1], order=[True, False], hue_order=[True, False])
ax[1].set_ylabel('')
ax[1].set_title('Count and Comparison w/Target')
fig.suptitle('Solo Travellers\n')

print(data['HomePlanet'].value_counts(dropna=False), '\n')
(fig, ax) = plt.subplots(1, 2, figsize=(12, 2))
data['HomePlanet'].value_counts(dropna=False).plot(kind='pie', autopct='%.1f%%', ax=ax[0], startangle=90, cmap='Set2', textprops={'fontsize': 8})
ax[0].set_ylabel('')
ax[0].set_title('Distribution')
sns.countplot(y=data['HomePlanet'], order=data['HomePlanet'].value_counts().index, ax=ax[1], hue=data['Transported'], hue_order=[True, False])
ax[1].set_ylabel('')
ax[1].set_xlabel('')
ax[1].set_title('Count and Comparison w/Target')
fig.suptitle('HomePlanet')

print(data['CryoSleep'].value_counts(dropna=False).sort_index(ascending=False), '\n')
(fig, ax) = plt.subplots(1, 2, figsize=(12, 2))
data['CryoSleep'].value_counts(dropna=False).sort_index(ascending=False).plot(kind='pie', cmap='Set2', autopct='%.1f%%', ax=ax[0], startangle=90, textprops={'fontsize': 8})
ax[0].set_ylabel('')
ax[0].set_title('Distribution')
sns.countplot(y=data['CryoSleep'], hue=data['Transported'], ax=ax[1], hue_order=[True, False])
ax[1].set_ylabel('')
ax[1].set_xlabel('')
ax[1].set_title('Count and Comparison w/Target')
fig.suptitle('CryoSleep')

print('Total VIP Passengers:', data[data.VIP == True].shape[0])
(fig, ax) = plt.subplots(1, 1, figsize=(5, 2))
data[data.VIP == True].CryoSleep.value_counts().sort_index(ascending=False).plot(kind='pie', autopct='%.1f%%', cmap='Set2', startangle=90, textprops={'fontsize': 8})
ax.set_ylabel('')
plt.suptitle('CryoSleep')

(fig, ax) = plt.subplots(1, 1, figsize=(10, 3))
data[data.CryoSleep == True].Age.value_counts().sort_index().plot(title='Count Passengers by Age', color='blue', label='Cryo Passengers')
data[data.CryoSleep == False].Age.value_counts().sort_index().plot(color='red', label='Non-Cryo Passengers')
plt.legend()

(fig, ax) = plt.subplots(1, 1, figsize=(10, 3))
data[data.CryoSleep == True].GroupSize.value_counts().sort_index().plot(title='blue Passengers by GroupSize', color='blue', label='Cryo Passengers')
data[data.CryoSleep == False].GroupSize.value_counts().sort_index().plot(title='Count Passengers by GroupSize', color='red', label='Non-Cryo Passengers')
plt.yticks(np.arange(0, 5000, 1000))
plt.legend()

print('Nulls Count Cabin:', data[data.Cabin.isnull()].shape[0], '\n')
print(data['Cabin'].head())
GroupId_DIM = data[['GroupId', 'Cabin']].drop_duplicates(subset='GroupId').set_index('GroupId').squeeze()

data.Cabin.fillna('GroupId_DIM')
data.loc[data.Cabin.isnull(), 'Cabin'] = pd.merge(left=data.loc[data.Cabin.isnull()][['GroupId', 'Cabin']], right=GroupId_DIM, how='left', left_on='GroupId', right_index=True)['Cabin_y']
print('Nulls Count Cabin:', data[data.Cabin.isnull()].shape[0], '\n')
col_names = ['Deck', 'Num', 'Side']
data_Cabin = data['Cabin'].str.split('/', expand=True).set_axis(col_names, axis=1)
data_Cabin.loc[data_Cabin.Num.notnull(), 'Num'] = data_Cabin.loc[data_Cabin.Num.notnull()].Num.astype(int)

data = pd.concat([data, data_Cabin], axis=1)
print(data['Deck'].value_counts(dropna=False))
(fig, ax) = plt.subplots(1, 1, figsize=(8, 3))
sns.countplot(y=data['Deck'], palette='Set2', order=data['Deck'].value_counts().index)
ax.set_ylabel('')
ax.set_xlabel('')
plt.title('Count by Deck')

print(data.Side.value_counts(dropna=False), '\n')
(fix, ax) = plt.subplots(1, 2, figsize=(10, 2))
data['Side'].value_counts(dropna=False).sort_index(ascending=False).plot(kind='pie', autopct='%.1f%%', ax=ax[0], startangle=270, cmap='Set2', textprops={'fontsize': 8})
ax[0].set_ylabel('')
ax[0].set_title('Distribution')
sns.countplot(y=data['Side'], hue=data['Transported'], ax=ax[1], hue_order=[True, False])
ax[1].set_ylabel('')
ax[1].set_xlabel('')
ax[1].set_title('Count and Comparison w/Target')

print('Nulls for Deck, Side and Num are the same:', data[data.Deck.isnull() & data.Side.isnull() & data.Num.isnull()].shape[0])
print('Nulls:', data['Num'].isnull().sum(), '\n')
Cabin_P_F = data[(data.Side == 'P') & (data.Deck == 'F')].groupby(['Side', 'Deck', 'Num']).size().to_frame('Count')
Cabin_S_F = data[(data.Side == 'S') & (data.Deck == 'F')].groupby(['Side', 'Deck', 'Num']).size().to_frame('Count')


print(data['Destination'].value_counts(dropna=False), '\n')
(fig, ax) = plt.subplots(1, 2, figsize=(12, 2))
data['Destination'].value_counts(dropna=False).plot(kind='pie', autopct='%.1f%%', ax=ax[0], startangle=90, cmap='Set2', textprops={'fontsize': 8})
ax[0].set_ylabel('')
ax[0].set_title('Distribution')
sns.countplot(y=data['Destination'], order=data['Destination'].value_counts().index, ax=ax[1], hue=data['Transported'])
ax[1].set_ylabel('')
ax[1].set_title('Count and Comparison w/Target')
plt.suptitle('Destination')

print('Nulls:', data['Age'].isnull().sum())
print('Skewness: %f' % data['Age'].skew(), '\n')
print(data['Age'].describe())
(fig, ax) = plt.subplots(1, 1, figsize=(18, 4))
sns.histplot(data=data['Age'], binwidth=5)
plt.title('Age')

print(data['VIP'].value_counts(dropna=False).sort_index(ascending=False), '\n')
(fig, ax) = plt.subplots(1, 2, figsize=(12, 2))
data['VIP'].value_counts(dropna=False).plot(kind='pie', autopct='%.1f%%', startangle=90, cmap='Set2', ax=ax[0], textprops={'fontsize': 8})
ax[0].set_ylabel('')
ax[0].set_title('Distribution')
sns.countplot(y=data['VIP'], hue=data['Transported'], order=[True, False], hue_order=[True, False], ax=ax[1])
ax[1].set_ylabel('')
ax[1].set_title('Count and Comparison w/Target')
fig.suptitle('VIP')

print('Nulls:', data['Name'].isnull().sum(), '\n')
print(data['Name'].value_counts())
print('Nulls:', data['RoomService'].isnull().sum(), '\n')
print(data['RoomService'].describe())
(fig, ax) = plt.subplots(2, 1, figsize=(10, 2))
sns.boxplot(x=data['RoomService'], showfliers=True, ax=ax[0])
ax[0].set_ylabel('')
sns.boxplot(x=data['RoomService'], showfliers=False, ax=ax[1])
ax[1].set_ylabel('')
fig.suptitle('RoomService with and without Outliers')

print('Nulls:', data['FoodCourt'].isnull().sum(), '\n')
print(data['FoodCourt'].describe())
(fig, ax) = plt.subplots(2, 1, figsize=(8, 2))
sns.boxplot(x=data['FoodCourt'], showfliers=True, ax=ax[0])
ax[0].set_ylabel('')
sns.boxplot(x=data['FoodCourt'], showfliers=False, ax=ax[1])
ax[1].set_ylabel('')
fig.suptitle('FoodCourt with and without Outliers')

print('Nulls:', data['ShoppingMall'].isnull().sum(), '\n')
print(data['ShoppingMall'].describe())
(fig, ax) = plt.subplots(2, 1, figsize=(8, 2))
sns.boxplot(x=data['ShoppingMall'], showfliers=True, ax=ax[0])
ax[0].set_ylabel('')
sns.boxplot(x=data['ShoppingMall'], showfliers=False, ax=ax[1])
ax[1].set_ylabel('')
fig.suptitle('ShoppingMall with and without Outliers')

print('Nulls:', data['Spa'].isnull().sum(), '\n')
print(data['Spa'].describe())
(fig, ax) = plt.subplots(2, 1, figsize=(8, 2))
sns.boxplot(x=data['Spa'], showfliers=True, ax=ax[0])
ax[0].set_ylabel('')
sns.boxplot(x=data['Spa'], showfliers=False, ax=ax[1])
ax[1].set_ylabel('')
fig.suptitle('Spa with and without Outliers')

print('Nulls:', data['VRDeck'].isnull().sum(), '\n')
print(data['VRDeck'].describe())
(fig, ax) = plt.subplots(2, 1, figsize=(8, 2))
sns.boxplot(x=data['VRDeck'], showfliers=True, ax=ax[0])
ax[0].set_ylabel('')
sns.boxplot(x=data['VRDeck'], showfliers=False, ax=ax[1])
ax[1].set_ylabel('')
fig.suptitle('VRDeck with and without Outliers')

data_plot = pd.concat([data.select_dtypes(['float64']).fillna(0), data['Transported']], axis=1)
sns.pairplot(data=data_plot, hue='Transported', hue_order=[True, False])

print('Nulls Count:', data[data.HomePlanet.isnull()].shape[0], '\n')

print('In groups with other passengers', data[data.HomePlanet.isnull() & ~data.IsAlone].shape[0])
data_grp_groupid_homeplanet = data.groupby(['GroupId', 'HomePlanet']).size().to_frame('Count').reset_index()

print('Is GroupId unique?:', data_grp_groupid_homeplanet.GroupId.is_unique)
data.loc[:, 'HomePlanet'] = data.sort_values(by=['GroupId', 'HomePlanet'])['HomePlanet'].fillna(method='ffill')
print('Nulls Count HomePlanet:', data.HomePlanet.isnull().sum())
print('Nulls Count CryoSleep:', data[data.CryoSleep.isnull()].shape[0], '\n')
mask_amenities_expenses = (data.RoomService > 0) | (data.FoodCourt > 0) | (data.ShoppingMall > 0) | (data.VRDeck > 0)
data.loc[mask_amenities_expenses & data.CryoSleep.isnull(), 'CryoSleep'] = False
print('Nulls Count CryoSleep:', data[data.CryoSleep.isnull()].shape[0], '\n')

data.loc[data.CryoSleep.isnull() & data.Transported, 'CryoSleep'] = True
print('\nNulls Count CryoSleep:', data[data.CryoSleep.isnull()].shape[0], '\n')
print(data[data.CryoSleep.isnull()].VIP.value_counts())
data.loc[data.CryoSleep.isnull() & data.VIP, 'CryoSleep'] = False
print('\nNulls Count CryoSleep:', data[data.CryoSleep.isnull()].shape[0], '\n')
data['CryoSleep'].fillna(data['CryoSleep'].mode()[0], inplace=True)
print('\nNulls Count CryoSleep:', data[data.CryoSleep.isnull()].shape[0], '\n')
print('Nulls Count Destination:', data[data.Destination.isnull()].shape[0], '\n')
data['Destination'].fillna(data['Destination'].mode()[0], inplace=True)
print('Nulls Count Destination:', data[data.Destination.isnull()].shape[0], '\n')
print('Nulls Count Age:', data[data.Age.isnull()].shape[0], '\n')
print('Nulls Count VIP:', data[data.VIP.isnull()].shape[0], '\n')
data['VIP'].fillna(data['VIP'].mode()[0], inplace=True)
print('Nulls Count VIP:', data[data.VIP.isnull()].shape[0], '\n')
continuous_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'Age']
print('Nulls Count Amenities:', data[continuous_cols].isnull().sum())
for col in continuous_cols:
    data[col].fillna(data[col].median(), inplace=True)
print('Nulls Count Amenities:', data[continuous_cols].isnull().sum())
print('Nulls Count Deck:', data[data.Deck.isnull()].shape[0], '\n')
data.dropna(subset=['Deck'], inplace=True)
print('Nulls Count Deck:', data[data.Deck.isnull()].shape[0], '\n')
X_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'GroupId', 'GroupSize', 'IsAlone', 'Deck', 'Num', 'Side']
X = data[X_cols]
y = data['Transported']
print(X.info())
print('Target size:', y.shape)
nominal_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Side', 'Deck']
X_nominals = data[nominal_cols].astype('category').reset_index(drop=True)
print(X_nominals.info())
from sklearn.preprocessing import OneHotEncoder
oh_encoder = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse=False, drop='first')
oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first')
X_nominals_encoded = oh_encoder.fit_transform(X_nominals)
X_nominals_encoded = pd.DataFrame(X_nominals_encoded, columns=oh_encoder.get_feature_names_out())

ordinal_cols = ['GroupSize']
X_ordinals = data[ordinal_cols].astype('category').reset_index(drop=True)
print(X_ordinals.info())
from sklearn.preprocessing import OrdinalEncoder
ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
X_ordinals_encoded = ord_encoder.fit_transform(X_ordinals)
X_ordinals_encoded = pd.DataFrame(X_ordinals_encoded, columns=X_ordinals.columns.values)

X_continuous = data[continuous_cols].reset_index(drop=True)
print(X_continuous.info())
from sklearn.preprocessing import KBinsDiscretizer
kbins_encoder = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
X_continuous_encoded = kbins_encoder.fit_transform(X_continuous)
X_continuous_encoded = pd.DataFrame(X_continuous_encoded, columns=X_continuous.columns)

from sklearn.preprocessing import LabelEncoder
lb_encoder = LabelEncoder()
y_encoded = lb_encoder.fit_transform(data['Transported'])
y_encoded = pd.Series(y_encoded, name='Transported')
X_encoded = pd.concat([X_nominals_encoded, X_ordinals_encoded, X_continuous], axis=1)
data_tmp = pd.concat([X_encoded, y_encoded], axis=1)
sns.heatmap(data_tmp.corr(), vmin=-1, vmax=1, cmap='RdYlGn')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
from sklearn.preprocessing import StandardScaler