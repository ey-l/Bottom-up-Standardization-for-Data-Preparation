import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
print(f'{train.head()}\n{train.info()}')
print(f'Training NaNs:\n{train.isnull().sum()}\n\nTesting NaNs:\n{test.isnull().sum()}')
print(f'\nThe data contains {train.isnull().sum().sum() + test.isnull().sum().sum()} NaNs')
(fig1, ax) = plt.subplots(2, 2, figsize=(10, 10))
idx = 0
for row in [0, 1]:
    for col in [0, 1]:
        feat = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP'][idx]
        train[feat].value_counts(dropna=False).plot(kind='pie', ax=ax[row][col])
        idx += 1
original_numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig2, ax) = plt.subplots(2, 3, figsize=(10, 10))
train[original_numerical_features].hist(ax=ax)
contains_nans = train.columns[train.isnull().any()]
for col in train.isnull().sum().index[0:-1]:
    if train[col].isnull().sum() > 0 and train[col].dtypes == 'float64':
        temp = train[col].median()
    else:
        temp = train[col].value_counts().index[0]
    train[col] = train[col].fillna(temp)
    test[col] = test[col].fillna(temp)
print(f'Training NaNs:\n{train.isnull().sum()}\n\nTesting NaNs:\n{test.isnull().sum()}')
print(f'\nThe data contains {train.isnull().sum().sum() + test.isnull().sum().sum()} NaNs')
train[['PassengerId_0', 'PassengerId_1']] = train['PassengerId'].str.split('_', 1, expand=True)
test[['PassengerId_0', 'PassengerId_1']] = test['PassengerId'].str.split('_', 1, expand=True)
train[['Deck', 'Number', 'Side']] = train['Cabin'].str.split('/', 2, expand=True)
test[['Deck', 'Number', 'Side']] = test['Cabin'].str.split('/', 2, expand=True)
train[['First name', 'Family name']] = train['Name'].str.split(' ', 1, expand=True)
test[['First name', 'Family name']] = test['Name'].str.split(' ', 1, expand=True)
train.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
test.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
train['Food/drink'] = train['RoomService'] + train['FoodCourt']
test['Food/drink'] = test['RoomService'] + test['FoodCourt']
train['Entertainment'] = train['Spa'] + train['VRDeck']
test['Entertainment'] = test['Spa'] + test['VRDeck']
train['TotalSpent'] = train['RoomService'] + train['FoodCourt'] + train['ShoppingMall'] + train['Spa'] + train['VRDeck']
test['TotalSpent'] = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']

def is_baller(df):
    mask = (df['CryoSleep'] == False) & (df['TotalSpent'] > df['TotalSpent'].median())
    return mask
train['Baller'] = is_baller(train)
test['Baller'] = is_baller(test)
danger_decks = []
for (idx, deck) in enumerate(train['Deck'].unique()):
    temp = train[train['Deck'] == deck]
    counts = temp['Transported'].value_counts().sort_index(ascending=False).values
    if counts[1] > counts[0]:
        print(f'{deck} is a dangerous deck. {counts[1]} people died compared to the {counts[0]} that survived')
        if deck not in danger_decks:
            danger_decks.append(deck)
train['DangerDeck'] = np.where(train['Deck'].isin(danger_decks), 1, 0)
test['DangerDeck'] = np.where(test['Deck'].isin(danger_decks), 1, 0)
numerical_features = []
categorical_features = []
for col in train.columns:
    if train[col].dtypes == 'float64' and col not in numerical_features:
        numerical_features.append(col)
    elif col != 'Transported' and col not in categorical_features:
        categorical_features.append(col)
print(f'There are {len(numerical_features)} numerical features and {len(categorical_features)} categorical features')

def log_transform(df, col):
    transformed = (df[col] - df[col].min() + 1).transform(np.log)
    return transformed
for (idx, feat) in enumerate(numerical_features):
    transformed_feat = log_transform(train, feat)
    correlation = transformed_feat.corr(train['Transported'])
    if abs(correlation) > 0.3:
        name = 'Log' + feat
        train[name] = transformed_feat
        test[name] = log_transform(test, feat)
        print(f'{feat} {correlation}')
print(f'Training:\n{train.head()}\nTesting:\n{test.head()}')
train.info()
from sklearn.preprocessing import MinMaxScaler, StandardScaler
min_max_scaler = MinMaxScaler()
stand_scaler = StandardScaler()
train[numerical_features] = min_max_scaler.fit_transform(train[numerical_features])
test[numerical_features] = min_max_scaler.fit_transform(test[numerical_features])
from sklearn.preprocessing import LabelEncoder

def encode_df_cols(df, columns):
    for col in columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype('str'))
    return df
train = encode_df_cols(train, categorical_features)
test = encode_df_cols(test, categorical_features)
print(f'Training:\n{train.head()}\nTesting:\n{test.head()}')
train.isnull().sum()
for col in train.columns:
    if train[col].dtypes == 'float64' and col not in numerical_features:
        numerical_features.append(col)
    elif col != 'Transported' and col not in categorical_features:
        categorical_features.append(col)
import seaborn as sns
(fig, ax) = plt.subplots(3, 3, figsize=(15, 15))
idx = 0
for row in range(0, 3):
    for col in range(0, 3):
        sns.kdeplot(data=train, x=numerical_features[idx], hue='Transported', ax=ax[row][col])
        idx += 1
(fig, ax) = plt.subplots(figsize=(15, 10))
numerical_corrs = train[numerical_features + ['Transported']].astype(float).corr()
sns.heatmap(numerical_corrs, annot=True, ax=ax)
print(numerical_corrs['Transported'].abs().sort_values(ascending=False))
(fig, ax) = plt.subplots(figsize=(15, 10))
categorical_corrs = train[categorical_features + ['Transported']].corr()
sns.heatmap(categorical_corrs, annot=True)
print(categorical_corrs['Transported'].abs().sort_values(ascending=False))
total_corrs = pd.concat([numerical_corrs['Transported'], categorical_corrs['Transported']]).abs().sort_values(ascending=False)
print(total_corrs)
corr_threshold = 0.15
total_corrs[total_corrs > corr_threshold][1:]
train = train[total_corrs[total_corrs > corr_threshold][1:].keys()]
test = test[total_corrs[total_corrs > corr_threshold][2:].keys()]
print(f'Training:\n{train.head()}\nTesting:\n{test.head()}')
Y_train = train['Transported']
X_train = train.loc[:, train.columns != 'Transported']
from sklearn.neural_network import MLPClassifier
neural_net = MLPClassifier(hidden_layer_sizes=(4000, 200, 100, 20), early_stopping=True, validation_fraction=0.2, verbose=True)