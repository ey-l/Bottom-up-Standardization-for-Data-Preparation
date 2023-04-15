import pandas as pd
import numpy as np
import copy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
titanic_train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
titanic_train_df.shape
titanic_train_df.info()
titanic_train_df.head()
plt.bar(titanic_train_df.columns, titanic_train_df.isna().sum())
plt.xticks(rotation=90)

plt.figure(figsize=(15, 10))
sns.heatmap(titanic_train_df.loc[:, ~titanic_train_df.columns.isin(['PassengerId', 'Transported'])].isna(), cmap='YlGnBu')

msno.heatmap(titanic_train_df, figsize=(15, 10))
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
plt.figure(figsize=(15, 15))
plt.subplots_adjust(hspace=0.25)
for i in range(0, len(numerical_cols)):
    plt.subplot(3, 2, i + 1)
    plt.title(numerical_cols[i])
    plt.hist(titanic_train_df.loc[:, numerical_cols[i]])

def cabin_transform(cabin, index):
    if cabin is np.nan:
        return cabin
    else:
        return str(cabin).split('/')[index]

def modify_features(orig_df):
    df = copy.deepcopy(orig_df)
    df.insert(0, 'PassengerGroup', df['PassengerId'].transform(lambda passengerId: int(passengerId.split('_')[0])))
    df.insert(1, 'GroupCount', df.groupby('PassengerGroup')['PassengerId'].transform('count'))
    df['Deck'] = df['Cabin'].transform(lambda cabin: cabin_transform(cabin, 0))
    df['DeckNumber'] = df['Cabin'].transform(lambda cabin: cabin_transform(cabin, 1))
    df['DeckSide'] = df['Cabin'].transform(lambda cabin: cabin_transform(cabin, 2))
    df['FamilyName'] = df['Name'].transform(lambda name: name if name is np.nan else str(name).split(' ')[-1])
    return df

def bin_numerical_values(orig_df):
    df = copy.deepcopy(orig_df)
    ageLabels = ['children', 'youth', 'adult', 'senior']
    amountLabels = ['< 1000', '< 2000', '> 2000']
    df['Age'] = pd.cut(df['Age'], bins=[0, 15, 24, 64, np.inf], labels=ageLabels, include_lowest=True)
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = pd.cut(df[col], bins=[0, 1000, 2000, np.inf], labels=amountLabels, include_lowest=True)
    return df

def encode_output(y):
    encoder = LabelEncoder()
    return encoder.fit_transform(y)

def preprocess_training_data(orig_df):
    df = copy.deepcopy(orig_df)
    df = modify_features(df)
    y = encode_output(df['Transported'])
    df = df.drop(['PassengerGroup', 'PassengerId', 'Cabin', 'Name', 'Transported'], axis=1)
    cols = df.columns
    medianImputer = SimpleImputer(strategy='median')
    constantImputer = SimpleImputer(strategy='constant', fill_value='unknown')
    meanImputer = SimpleImputer(strategy='mean')
    df[['Age']] = meanImputer.fit_transform(df[['Age']])
    df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = medianImputer.fit_transform(df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
    df = constantImputer.fit_transform(df)
    imputers = {'constant': constantImputer, 'median': medianImputer, 'mean': meanImputer}
    df = pd.DataFrame(df, columns=cols)
    df = df.convert_dtypes()
    df = bin_numerical_values(df)
    encoder_models = {}
    for col in df.columns:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        if df[col].dtypes != 'object':
            df[[col]] = encoder.fit_transform(df[[col]].astype('category'))
        else:
            df[[col]] = encoder.fit_transform(df[[col]].astype('string'))
        encoder_models[col] = encoder
    return (df, y, imputers, encoder_models)
(train_df, y, imputers, encoders) = preprocess_training_data(titanic_train_df)
train_df.head()
train_df.isna().sum()

def preprocess_test_data(orig_df, imputers, encoders):
    df = copy.deepcopy(orig_df)
    df = modify_features(df)
    df = df.drop(['PassengerGroup', 'PassengerId', 'Cabin', 'Name'], axis=1)
    cols = df.columns
    df[['Age']] = imputers['mean'].transform(df[['Age']])
    df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = imputers['median'].transform(df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
    df = imputers['constant'].transform(df)
    df = pd.DataFrame(df, columns=cols)
    df = df.convert_dtypes()
    df = bin_numerical_values(df)
    for col in encoders:
        if df[col].dtypes != 'object':
            df[[col]] = encoders[col].transform(df[[col]].astype('category'))
        else:
            df[[col]] = encoders[col].transform(df[[col]].astype('string'))
    return df
titanic_test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df = preprocess_test_data(titanic_test_df, imputers, encoders)
test_df.head()
test_df.isna().sum()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
(train_X, test_X, train_y, test_y) = train_test_split(train_df, y, test_size=0.1, shuffle=True, random_state=0)
print('X train shape : ', train_X.shape)
print('y train shape : ', train_y.shape)
print('X test shape : ', test_X.shape)
print('y test shape : ', test_y.shape)
from sklearn.metrics import accuracy_score
clf = LogisticRegression(max_iter=2000)