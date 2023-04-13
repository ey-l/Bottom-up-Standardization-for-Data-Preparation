import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import missingno as mn
from scipy import stats
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler, power_transform
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from scipy import stats
sns.set_style('whitegrid')
sns.color_palette('flare')
sns.set_palette(sns.color_palette('flare'))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
test = _input0.copy()
enc = LabelEncoder()
_input1['Transported'] = enc.fit_transform(_input1['Transported'])
sns.countplot(data=_input1, x=_input1.Transported)
print('shape of train data:', _input1.shape)
print('shape of test data:', test.shape)
print('=' * 100)
num_col = [a for a in _input1.columns if _input1[a].dtype in ['float64']]
cat_col = [a for a in _input1.columns if _input1[a].dtype in ['object']]
print('\nNumerical_features', num_col)
print('\nCategorical_features', cat_col)
_input1.head()
y = _input1['Transported']
_input1.head()
print('=' * 100)

def fill_values(df):
    df['ShoppingMall'] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df[['ShoppingMall']])
    df['RoomService'] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df[['RoomService']])
    df['FoodCourt'] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df[['FoodCourt']])
    df['Spa'] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df[['Spa']])
    df['VRDeck'] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df[['VRDeck']])
    df['Age'] = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(df[['Age']])
    df['HomePlanet'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['HomePlanet']])
    df['CryoSleep'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['CryoSleep']])
    df['Cabin'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Cabin']])
    df['Destination'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Destination']])
    df['VIP'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['VIP']])
fill_values(_input1)
fill_values(test)
print(_input1.info())
print(test.info())
_input1[['Group_No', 'No_in_group']] = _input1['PassengerId'].str.split('_', expand=True)
test[['Group_No', 'No_in_group']] = test['PassengerId'].str.split('_', expand=True)
_input1['total_exp'] = _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck'] + _input1['FoodCourt'] + _input1['RoomService']
test['total_exp'] = test['ShoppingMall'] + test['Spa'] + test['VRDeck'] + test['FoodCourt'] + test['RoomService']
_input1[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = _input1['Cabin'].str.split('/', expand=True)
test[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = test['Cabin'].str.split('/', expand=True)
_input1 = _input1.drop(columns=['PassengerId', 'Name', 'Cabin'], axis=1, inplace=False)
test = test.drop(columns=['PassengerId', 'Name', 'Cabin'], axis=1, inplace=False)
cat_col = [a for a in _input1.columns if _input1[a].dtype in ['object']]
num_col = [a for a in _input1.columns if _input1[a].dtype in ['float64']]
print('Categorical features: ', cat_col)
print('Numerical features: ', num_col)
(fig, ax) = plt.subplots(3, 3, figsize=(20, 15))
sns.kdeplot(ax=ax[0, 0], x='Age', data=_input1, fill=True, hue=round(_input1['Age'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[0, 1], x='RoomService', data=_input1, fill=True, hue=round(_input1['RoomService'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[0, 2], x='FoodCourt', data=_input1, fill=True, hue=round(_input1['FoodCourt'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 0], x='ShoppingMall', data=_input1, fill=True, hue=round(_input1['ShoppingMall'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 1], x='Spa', data=_input1, fill=True, hue=round(_input1['Spa'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 2], x='VRDeck', data=_input1, fill=True, hue=round(_input1['VRDeck'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[2, 0], x='total_exp', data=_input1, fill=True, hue=round(_input1['total_exp'].skew(), 2), palette='Spectral')
_input1[num_col] = power_transform(_input1[num_col], method='yeo-johnson', standardize=True)
test[num_col] = power_transform(test[num_col], method='yeo-johnson', standardize=True)
(fig, ax) = plt.subplots(3, 3, figsize=(20, 15))
sns.kdeplot(ax=ax[0, 0], x='Age', data=_input1, fill=True, hue=round(_input1['Age'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[0, 1], x='RoomService', data=_input1, fill=True, hue=round(_input1['RoomService'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[0, 2], x='FoodCourt', data=_input1, fill=True, hue=round(_input1['FoodCourt'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 0], x='ShoppingMall', data=_input1, fill=True, hue=round(_input1['ShoppingMall'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 1], x='Spa', data=_input1, fill=True, hue=round(_input1['Spa'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 2], x='VRDeck', data=_input1, fill=True, hue=round(_input1['VRDeck'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[2, 0], x='total_exp', data=_input1, fill=True, hue=round(_input1['total_exp'].skew(), 2), palette='Spectral')
_input1[cat_col] = OrdinalEncoder().fit_transform(_input1[cat_col])
test[cat_col] = OrdinalEncoder().fit_transform(test[cat_col])
features = num_col + cat_col
X = _input1[features]
y = _input1['Transported']
cv = 5
metric = []
kfold = KFold(n_splits=cv, random_state=42, shuffle=True)
for (idx, (train_idx, val_idx)) in enumerate(kfold.split(_input1)):
    (xtrain, xvalid) = (X.loc[train_idx], X.loc[val_idx])
    (ytrain, yvalid) = (y.loc[train_idx], y.loc[val_idx])
    log_model = LogisticRegression(penalty='l2', C=10, max_iter=1000)