import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
titanic_train_path = 'data/input/spaceship-titanic/train.csv'
titanic_train_raw = pd.read_csv(titanic_train_path)
print(titanic_train_raw.info())
titanic_train_raw.head()
titanic_train_raw.isnull().sum()

def extract_info(df):
    df['GroupID'] = df['PassengerId'].str.split('_', expand=True)[0]
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['CabinDeck'] = cabin_split[0]
    df['CabinNum'] = cabin_split[1]
    df['CabinSide'] = cabin_split[2]
    df.drop('Cabin', inplace=True, axis=1)
    name_split = df['Name'].str.split(' ', expand=True)
    df['FamilyName'] = name_split[1]
    df.drop('Name', inplace=True, axis=1)
    amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalFee'] = df[amenities].sum(axis=1)
    numeric = ['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupID', 'CabinNum', 'TotalFee']
    for n in numeric:
        df[n] = pd.to_numeric(df[n])
    return df
titanic_train = titanic_train_raw.copy()
titanic_train = extract_info(titanic_train)
print(titanic_train.info())
titanic_train
list_group = ['FamilyName', 'GroupID']
list_target = ['HomePlanet', 'Destination']
for group in list_group:
    for target in list_target:
        num_target_in_group = titanic_train.groupby([group])[target].nunique().reset_index()
        print('# of %s with different %s: %d' % (group, target, num_target_in_group[num_target_in_group[target] > 1].shape[0]))
print('\ntotal # of families: %s  total # of groups: %s' % (titanic_train.nunique()['FamilyName'], titanic_train.nunique()['GroupID']))

def homeplanet_vs_group_family_fill(df):
    list_group = ['GroupID', 'FamilyName']
    for group in list_group:
        home_planet_in_group = df.dropna(subset=[group, 'HomePlanet']).drop_duplicates([group])[[group, 'HomePlanet']]
        home_planet_dict = dict(zip(home_planet_in_group[group], home_planet_in_group['HomePlanet']))
        df['HomePlanet'] = df['HomePlanet'].fillna(df[group].map(home_planet_dict))
    return df
titanic_train = homeplanet_vs_group_family_fill(titanic_train)

def cryo_vs_amenities_fill(df):
    df['RoomService'] = np.where(df['CryoSleep'] == True, 0, df['RoomService'])
    df['FoodCourt'] = np.where(df['CryoSleep'] == True, 0, df['FoodCourt'])
    df['ShoppingMall'] = np.where(df['CryoSleep'] == True, 0, df['ShoppingMall'])
    df['Spa'] = np.where(df['CryoSleep'] == True, 0, df['Spa'])
    df['VRDeck'] = np.where(df['CryoSleep'] == True, 0, df['VRDeck'])
    df['CryoSleep'] = np.where(df['TotalFee'] > 0, False, df['CryoSleep'])
    return df
titanic_train = cryo_vs_amenities_fill(titanic_train)
under_21 = titanic_train[titanic_train['Age'] < 21]
(fig, ax) = plt.subplots(1, 2, figsize=(18, 4))
ax[0].set_title('Age vs VIP')
sns.barplot(x=under_21.Age, y=under_21.VIP, ax=ax[0])
ax[1].set_title('Age vs Total Amenities Fee')
sns.barplot(x=under_21.Age, y=under_21.TotalFee, ax=ax[1])
fig.show()

def age_limit_fill(df, vip_limit=18, amenities_limit=13):
    df['VIP'] = np.where(df['Age'] < vip_limit, False, df['VIP'])
    df['RoomService'] = np.where(df['Age'] < amenities_limit, 0, df['RoomService'])
    df['FoodCourt'] = np.where(df['Age'] < amenities_limit, 0, df['FoodCourt'])
    df['ShoppingMall'] = np.where(df['Age'] < amenities_limit, 0, df['ShoppingMall'])
    df['Spa'] = np.where(df['Age'] < amenities_limit, 0, df['Spa'])
    df['VRDeck'] = np.where(df['Age'] < amenities_limit, 0, df['VRDeck'])
    df['TotalFee'] = np.where(df['Age'] < amenities_limit, 0, df['TotalFee'])
    return df
titanic_train = age_limit_fill(titanic_train)

def mode_by_group_n_family_fill(df, columns=['CabinDeck', 'CabinNum', 'CabinSide']):
    for col in columns:
        print(col)
        mode_by_group = df.dropna(subset=[col]).groupby(['GroupID', 'FamilyName'])[col].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
        mode_dict = dict(zip(mode_by_group.set_index(['GroupID', 'FamilyName']).index, mode_by_group[col]))
        df[col] = df[col].fillna(df.set_index(['GroupID', 'FamilyName']).index.map(mode_dict).to_series().reset_index(drop=True))
    return df
titanic_train = mode_by_group_n_family_fill(titanic_train)

def mode_by_group_fill(df, columns=['FamilyName', 'CabinDeck', 'CabinNum', 'CabinSide']):
    for col in columns:
        print(col)
        mode_by_group = df.dropna(subset=[col]).groupby('GroupID')[col].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
        mode_dict = dict(zip(mode_by_group['GroupID'], mode_by_group[col]))
        df[col] = df[col].fillna(df['GroupID'].map(mode_dict))
    return df
titanic_train = mode_by_group_fill(titanic_train)

def mode_fill(df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'VIP', 'FamilyName', 'CabinDeck', 'CabinNum', 'CabinSide']):
    for col in columns:
        print(col)
        df[col] = df[col].fillna(value=df[col].mode()[0])
    return df
titanic_train = mode_fill(titanic_train)

def mean_fill(df, columns=['Age']):
    for col in columns:
        print(col)
        df[col] = df[col].fillna(value=df[col].mean())
    return df
titanic_train = mean_fill(titanic_train)
titanic_train.isnull().sum()

def count_group_family(df):
    num_family_members = df.groupby('FamilyName')['PassengerId'].count().reset_index()
    num_family_members_dict = dict(zip(num_family_members['FamilyName'], num_family_members['PassengerId']))
    df['NumFamilyMembers'] = df['FamilyName'].map(num_family_members_dict)
    num_group_members = df.groupby('GroupID')['PassengerId'].count().reset_index()
    num_group_members_dict = dict(zip(num_group_members['GroupID'], num_group_members['PassengerId']))
    df['NumGroupMembers'] = df['GroupID'].map(num_group_members_dict)
    return df
titanic_train = count_group_family(titanic_train)

def age_group(df):
    df['Age_group'] = df['Age'] // 20
    return df
titanic_train = age_group(titanic_train)
titanic_train.isnull().sum()

def one_hot(df, columns):
    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=columns)
    return df
from sklearn.preprocessing import LabelEncoder

def preprocessing(train, test):
    train = extract_info(train)
    train = homeplanet_vs_group_family_fill(train)
    train = cryo_vs_amenities_fill(train)
    train = age_limit_fill(train)
    train = mode_by_group_n_family_fill(train)
    train = mode_by_group_fill(train)
    train = mode_fill(train)
    train = mean_fill(train)
    trian = count_group_family(train)
    train = age_group(train)
    test = extract_info(test)
    test = homeplanet_vs_group_family_fill(test)
    test = cryo_vs_amenities_fill(test)
    test = age_limit_fill(test)
    test = mode_by_group_n_family_fill(test)
    test = mode_by_group_fill(test)
    test = mode_fill(test)
    test = mean_fill(test)
    test = count_group_family(test)
    test = age_group(test)
    remove_col = ['PassengerId']
    one_hot_col = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide']
    label_encode_col = ['FamilyName']
    train = train.drop(columns=remove_col)
    test = test.drop(columns=remove_col)
    train = one_hot(train, one_hot_col)
    test = one_hot(test, one_hot_col)
    concat_data = pd.concat([train[label_encode_col], test[label_encode_col]])
    label_encoder = LabelEncoder()