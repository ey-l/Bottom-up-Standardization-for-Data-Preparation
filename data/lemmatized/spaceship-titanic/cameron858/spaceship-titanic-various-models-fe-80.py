import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
print(f'{_input1.head()}\n{_input1.info()}')
print(f'Training NaNs:\n{_input1.isnull().sum()}\n\nTesting NaNs:\n{_input0.isnull().sum()}')
print(f'\nThe data contains {_input1.isnull().sum().sum() + _input0.isnull().sum().sum()} NaNs')
(fig1, ax) = plt.subplots(2, 2, figsize=(10, 10))
idx = 0
for row in [0, 1]:
    for col in [0, 1]:
        feat = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP'][idx]
        _input1[feat].value_counts(dropna=False).plot(kind='pie', ax=ax[row][col])
        idx += 1
original_numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig2, ax) = plt.subplots(2, 3, figsize=(10, 10))
_input1[original_numerical_features].hist(ax=ax)
contains_nans = _input1.columns[_input1.isnull().any()]
for col in _input1.isnull().sum().index[0:-1]:
    if _input1[col].isnull().sum() > 0 and _input1[col].dtypes == 'float64':
        temp = _input1[col].median()
    else:
        temp = _input1[col].value_counts().index[0]
    _input1[col] = _input1[col].fillna(temp)
    _input0[col] = _input0[col].fillna(temp)
print(f'Training NaNs:\n{_input1.isnull().sum()}\n\nTesting NaNs:\n{_input0.isnull().sum()}')
print(f'\nThe data contains {_input1.isnull().sum().sum() + _input0.isnull().sum().sum()} NaNs')
_input1[['PassengerId_0', 'PassengerId_1']] = _input1['PassengerId'].str.split('_', 1, expand=True)
_input0[['PassengerId_0', 'PassengerId_1']] = _input0['PassengerId'].str.split('_', 1, expand=True)
_input1[['Deck', 'Number', 'Side']] = _input1['Cabin'].str.split('/', 2, expand=True)
_input0[['Deck', 'Number', 'Side']] = _input0['Cabin'].str.split('/', 2, expand=True)
_input1[['First name', 'Family name']] = _input1['Name'].str.split(' ', 1, expand=True)
_input0[['First name', 'Family name']] = _input0['Name'].str.split(' ', 1, expand=True)
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
_input0 = _input0.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
_input1['Food/drink'] = _input1['RoomService'] + _input1['FoodCourt']
_input0['Food/drink'] = _input0['RoomService'] + _input0['FoodCourt']
_input1['Entertainment'] = _input1['Spa'] + _input1['VRDeck']
_input0['Entertainment'] = _input0['Spa'] + _input0['VRDeck']
_input1['TotalSpent'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['TotalSpent'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']

def is_baller(df):
    mask = (df['CryoSleep'] == False) & (df['TotalSpent'] > df['TotalSpent'].median())
    return mask
_input1['Baller'] = is_baller(_input1)
_input0['Baller'] = is_baller(_input0)
danger_decks = []
for (idx, deck) in enumerate(_input1['Deck'].unique()):
    temp = _input1[_input1['Deck'] == deck]
    counts = temp['Transported'].value_counts().sort_index(ascending=False).values
    if counts[1] > counts[0]:
        print(f'{deck} is a dangerous deck. {counts[1]} people died compared to the {counts[0]} that survived')
        if deck not in danger_decks:
            danger_decks.append(deck)
_input1['DangerDeck'] = np.where(_input1['Deck'].isin(danger_decks), 1, 0)
_input0['DangerDeck'] = np.where(_input0['Deck'].isin(danger_decks), 1, 0)
numerical_features = []
categorical_features = []
for col in _input1.columns:
    if _input1[col].dtypes == 'float64' and col not in numerical_features:
        numerical_features.append(col)
    elif col != 'Transported' and col not in categorical_features:
        categorical_features.append(col)
print(f'There are {len(numerical_features)} numerical features and {len(categorical_features)} categorical features')

def log_transform(df, col):
    transformed = (df[col] - df[col].min() + 1).transform(np.log)
    return transformed
for (idx, feat) in enumerate(numerical_features):
    transformed_feat = log_transform(_input1, feat)
    correlation = transformed_feat.corr(_input1['Transported'])
    if abs(correlation) > 0.3:
        name = 'Log' + feat
        _input1[name] = transformed_feat
        _input0[name] = log_transform(_input0, feat)
        print(f'{feat} {correlation}')
print(f'Training:\n{_input1.head()}\nTesting:\n{_input0.head()}')
_input1.info()
from sklearn.preprocessing import MinMaxScaler, StandardScaler
min_max_scaler = MinMaxScaler()
stand_scaler = StandardScaler()
_input1[numerical_features] = min_max_scaler.fit_transform(_input1[numerical_features])
_input0[numerical_features] = min_max_scaler.fit_transform(_input0[numerical_features])
from sklearn.preprocessing import LabelEncoder

def encode_df_cols(df, columns):
    for col in columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype('str'))
    return df
_input1 = encode_df_cols(_input1, categorical_features)
_input0 = encode_df_cols(_input0, categorical_features)
print(f'Training:\n{_input1.head()}\nTesting:\n{_input0.head()}')
_input1.isnull().sum()
for col in _input1.columns:
    if _input1[col].dtypes == 'float64' and col not in numerical_features:
        numerical_features.append(col)
    elif col != 'Transported' and col not in categorical_features:
        categorical_features.append(col)
import seaborn as sns
(fig, ax) = plt.subplots(3, 3, figsize=(15, 15))
idx = 0
for row in range(0, 3):
    for col in range(0, 3):
        sns.kdeplot(data=_input1, x=numerical_features[idx], hue='Transported', ax=ax[row][col])
        idx += 1
(fig, ax) = plt.subplots(figsize=(15, 10))
numerical_corrs = _input1[numerical_features + ['Transported']].astype(float).corr()
sns.heatmap(numerical_corrs, annot=True, ax=ax)
print(numerical_corrs['Transported'].abs().sort_values(ascending=False))
(fig, ax) = plt.subplots(figsize=(15, 10))
categorical_corrs = _input1[categorical_features + ['Transported']].corr()
sns.heatmap(categorical_corrs, annot=True)
print(categorical_corrs['Transported'].abs().sort_values(ascending=False))
total_corrs = pd.concat([numerical_corrs['Transported'], categorical_corrs['Transported']]).abs().sort_values(ascending=False)
print(total_corrs)
corr_threshold = 0.15
total_corrs[total_corrs > corr_threshold][1:]
_input1 = _input1[total_corrs[total_corrs > corr_threshold][1:].keys()]
_input0 = _input0[total_corrs[total_corrs > corr_threshold][2:].keys()]
print(f'Training:\n{_input1.head()}\nTesting:\n{_input0.head()}')
Y_train = _input1['Transported']
X_train = _input1.loc[:, _input1.columns != 'Transported']
from sklearn.neural_network import MLPClassifier
neural_net = MLPClassifier(hidden_layer_sizes=(4000, 200, 100, 20), early_stopping=True, validation_fraction=0.2, verbose=True)