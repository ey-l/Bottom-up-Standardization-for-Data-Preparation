import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
import matplotlib.ticker as mtick
import time
import re
pd.set_option('float_format', '{:f}'.format)
import warnings
warnings.filterwarnings(action='ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train = _input1.copy()
test = _input0.copy()
sample_sub = _input2
train.info()
test.info()
train
train[['Group', 'Id']] = train['PassengerId'].str.split('_', expand=True)
test[['Group', 'Id']] = test['PassengerId'].str.split('_', expand=True)
train['Group'] = pd.to_numeric(train['Group'])
test['Group'] = pd.to_numeric(test['Group'])
train = train.drop(['Id', 'PassengerId'], inplace=False, axis=1)
test = test.drop(['Id', 'PassengerId'], inplace=False, axis=1)
train[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = train['Cabin'].str.split('/', expand=True)
test[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = test['Cabin'].str.split('/', expand=True)
train['Cabin_num'] = pd.to_numeric(train['Cabin_num'])
test['Cabin_num'] = pd.to_numeric(test['Cabin_num'])
train = train.drop('Cabin', inplace=False, axis=1)
test = test.drop('Cabin', inplace=False, axis=1)
train = train.drop('Name', inplace=False, axis=1)
test = test.drop('Name', inplace=False, axis=1)
Target = 'Transported'
Features = [col for col in train.columns]
cat_feat = [col for col in train.columns if (train[col].dtypes == 'object') & (col not in [Target])]
num_feat = [col for col in train.columns if (train[col].dtypes != 'object') & (col not in [Target])]
print("Train set's dimension : ", train.shape)
print("Test set's dimension : ", test.shape)
print('numbers of categorical feature : ', len(cat_feat))
print('numbers of numerical feature : ', len(num_feat))
print('Is there missing values?', train.isnull().sum().sum())
print('Is there missing values?', test.isnull().sum().sum())
print('train data missing columns\n\n', train.isnull().sum())
print('\n\ntest data missing columns\n\n', test.isnull().sum())
msno.matrix(train)
plt.title('Missing Value Distribution in train set')
msno.matrix(test)
plt.title('Missing Value Distribution in test set')
(fig, ax) = plt.subplots(1, 1, figsize=(12, 8))
(train.isnull().mean() * 100).plot(kind='bar', ax=ax, align='center', width=0.4, color='violet')
(test.isnull().mean() * 100).plot(kind='bar', ax=ax, align='edge', width=0.4, color='dodgerblue')
plt.legend(labels=['Train Set', 'Test Set'])
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.tick_params(axis='x', labelrotation=80)
ax.set_ylabel('Missing Values (%)')
ax.set_title('Percentage of missing values in train and test set')
plt.title('Distribution of Trasported')
sns.countplot(x='Transported', data=train)
print('Total number of Transported : ', len(train[train[Target] == True]))
print('Total number of Not transported : ', len(train[train[Target] == False]))
cat_feat
plt.figure(figsize=(20, 10))
plt.subplot(3, 2, 1)
plt.title('HomePlanet distribution based on Transported')
sns.countplot(x='HomePlanet', hue='Transported', data=train)
plt.subplot(3, 2, 2)
plt.title('CryoSleep distribution based on Transported')
sns.countplot(x='CryoSleep', hue='Transported', data=train)
plt.subplot(3, 2, 3)
plt.title('Destination distribution based on Transported')
sns.countplot(x='Destination', hue='Transported', data=train)
plt.subplot(3, 2, 4)
plt.title('VIP distribution based on Transported')
sns.countplot(x='VIP', hue='Transported', data=train)
plt.subplot(3, 2, 5)
plt.title('Cabin_deck distribution based on Transported')
sns.countplot(x='Cabin_deck', hue='Transported', data=train)
plt.subplot(3, 2, 6)
plt.title('Cabin_side distribution based on Transported')
sns.countplot(x='Cabin_side', hue='Transported', data=train)
plt.subplots_adjust(hspace=0.6)
num_feat
plt.figure(figsize=(20, 15))
plt.subplot(4, 2, 1)
plt.title('Age distribution based on Transported')
sns.histplot(x='Age', data=train, kde=True, alpha=0.1, color='#E6104C')
sns.histplot(x='Age', data=train, hue='Transported', multiple='stack')
plt.subplot(4, 2, 2)
plt.title('RoomService distribution based on Transported')
sns.histplot(x='RoomService', data=train, kde=True, alpha=0.1, color='#E6104C')
sns.histplot(x='RoomService', data=train, hue='Transported', multiple='stack')
plt.subplot(4, 2, 3)
plt.title('FoodCourt distribution based on Transported')
sns.histplot(x='FoodCourt', data=train, kde=True, alpha=0.1, color='#E6104C')
sns.histplot(x='FoodCourt', data=train, hue='Transported', multiple='stack')
plt.subplot(4, 2, 4)
plt.title('ShoppingMall distribution based on Transported')
sns.histplot(x='ShoppingMall', data=train, kde=True, alpha=0.1, color='#E6104C')
sns.histplot(x='ShoppingMall', data=train, hue='Transported', multiple='stack')
plt.subplot(4, 2, 5)
plt.title('Spa distribution based on Transported')
sns.histplot(x='Spa', data=train, kde=True, alpha=0.1, color='#E6104C')
sns.histplot(x='Spa', data=train, hue='Transported', multiple='stack')
plt.subplot(4, 2, 6)
plt.title('VRDeck distribution based on Transported')
sns.histplot(x='VRDeck', data=train, kde=True, alpha=0.1, color='#E6104C')
sns.histplot(x='VRDeck', data=train, hue='Transported', multiple='stack')
plt.subplot(4, 2, 7)
plt.title('Group distribution based on Transported')
sns.histplot(x='Group', data=train, kde=True, alpha=0.1, color='#E6104C')
sns.histplot(x='Group', data=train, hue='Transported', multiple='stack')
plt.subplot(4, 2, 8)
plt.title('Cabin_num distribution based on Transported')
sns.histplot(x='Cabin_num', data=train, kde=True, alpha=0.1, color='#E6104C')
sns.histplot(x='Cabin_num', data=train, hue='Transported', multiple='stack')
plt.subplots_adjust(hspace=0.6)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
Features = [col for col in train.columns if col not in ['PassengerId', Target]]
from impyute.imputation.cs import mice
train_imputed = mice(train.drop([Target], axis=1).values)
test_imputed = mice(test.values)
feat_colum_list = list(train.columns)
feat_colum_list.remove('Transported')
train[Features] = pd.DataFrame(train_imputed, columns=feat_colum_list)
test[Features] = pd.DataFrame(test_imputed, columns=feat_colum_list)
train[Features].head()
print('Is there missing values?', train.isnull().sum().sum())
print('Is there missing values?', test.isnull().sum().sum())
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
DTC = tree.DecisionTreeClassifier(random_state=0)
DTC_pred_result = []
DTC_scores = []
DTC_feature_imp = []
splitter = KFold(n_splits=10, shuffle=True, random_state=0)
print('Start Decision Tree Classify')
for (fold, (train_index, valid_index)) in enumerate(splitter.split(train[Features], train[Target])):
    print(10 * '=', f'Fold : {fold + 1}', 10 * '=')
    start_time = time.time()
    (X_train, X_valid) = (train.iloc[train_index][Features], train.iloc[valid_index][Features])
    (y_train, y_valid) = (train[Target].iloc[train_index], train[Target].iloc[valid_index])
    model = DTC