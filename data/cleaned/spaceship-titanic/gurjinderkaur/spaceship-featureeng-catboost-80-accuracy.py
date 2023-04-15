import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data.head()
train_data.shape
train_data.info()
train_data.isnull().sum()
categorical_features = train_data.select_dtypes('object').columns.to_list()
print(categorical_features)
numerical_features = train_data.drop(['Transported'], axis=1).select_dtypes(np.number).columns.to_list()
print(numerical_features)
spend_feature_list = train_data.drop(['Transported', 'Age'], axis=1).select_dtypes(np.number).columns.to_list()
for col in spend_feature_list:
    train_data[col] = train_data[col].fillna(0)
    test_data[col] = test_data[col].fillna(0)
from sklearn.impute import KNNImputer
train_data_num = train_data[numerical_features]
numerical_imputer = KNNImputer(n_neighbors=2)
train_num_transformed = numerical_imputer.fit_transform(train_data_num)
train_num_df = pd.DataFrame(train_num_transformed, columns=numerical_features, index=train_data.index)
print(train_num_df.head())
print('------------------')
print('Missing values:')
print(train_num_df.isnull().sum())
test_data_num = test_data[numerical_features]
test_num_transformed = numerical_imputer.fit_transform(test_data_num)
test_num_df = pd.DataFrame(test_num_transformed, columns=numerical_features, index=test_data.index)
print(test_num_df.head())
print('------------------')
print('Missing values:')
print(test_num_df.isnull().sum())
target = train_data['Transported']
train_cat_df = train_data[categorical_features]
train_df = pd.concat([train_cat_df, train_num_df], axis=1)
train_df['Transported'] = target
train_df
test_cat_df = test_data[categorical_features]
test_df = pd.concat([test_cat_df, test_num_df], axis=1)
test_df
train_df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True)
train_df['Transported'] = train_df['Transported'].astype(int)
sns.countplot(data=train_df, x='Transported')
(fig, ax) = plt.subplots(2, 2, figsize=(15, 10))
sns.countplot(x='CryoSleep', data=train_df, hue='Transported', ax=ax[0][0])
sns.countplot(x='VIP', data=train_df, hue='Transported', ax=ax[0][1])
sns.countplot(x='Destination', data=train_df, hue='Transported', ax=ax[1][0])
sns.countplot(x='HomePlanet', data=train_df, hue='Transported', ax=ax[1][1])
plt.figure(figsize=(8, 6))
sns.histplot(data=train_df, x='Age')
plt.figure(figsize=(8, 6))
sns.boxplot(data=train_df, x='Transported', y='Age')
plt.figure(figsize=(8, 6))
sns.histplot(data=train_df, x='Age', hue='Transported', multiple='stack')

def age_categorize(row):
    if row['Age'] >= 0 and row['Age'] < 13:
        return 'Child'
    elif row['Age'] >= 13 and row['Age'] < 20:
        return 'Teen'
    elif row['Age'] >= 20 and row['Age'] < 35:
        return 'Young Adult'
    elif row['Age'] >= 35 and row['Age'] < 55:
        return 'Middle-aged Adult'
    else:
        return 'Elderly'
train_df['AgeGroup'] = train_df.apply(lambda row: age_categorize(row), axis=1)
test_df['AgeGroup'] = test_df.apply(lambda row: age_categorize(row), axis=1)
plt.figure(figsize=(8, 6))
sns.histplot(data=train_df, x='AgeGroup', hue='Transported', multiple='stack', shrink=0.8)
train_df['PassengerId']
train_df[['Group', 'PersonNumber']] = train_df['PassengerId'].str.split('_', expand=True)
test_df[['Group', 'PersonNumber']] = test_df['PassengerId'].str.split('_', expand=True)
train_df['Group'].value_counts()
train_df['Name'].isnull().sum()
missing_names = train_df[train_df['Name'].isna()][['HomePlanet', 'Destination', 'Group', 'Name']]
missing_names
missing_names.groupby('HomePlanet').count()

def find_passenger(ind, hp, dst, grp, df):
    name = df[(df['HomePlanet'] == hp) & (df['Group'] == grp) | (df['Destination'] == dst) & (df['Group'] == grp)]['Name']
    return name.tolist()
for (index, row) in missing_names.iterrows():
    homeplanet = row['HomePlanet']
    dest = row['Destination']
    group = row['Group']
    name_list = find_passenger(index, homeplanet, dest, group, train_df)
    for i in name_list:
        if type(i) == float:
            row['Name'] = np.nan
        else:
            row['Name'] = 'Dummy ' + i.split()[1]
missing_names
missing_names.isna().sum()

def find_passenger_with_group(ind, group, df):
    name = df[df['Group'] == group]['Name']
    return name.tolist()
for (index, row) in missing_names.iterrows():
    if type(row['Name']) == float:
        group = row['Group']
        name_list = find_passenger_with_group(index, group, train_df)
        for i in name_list:
            if type(i) == float:
                row['Name'] = np.nan
            else:
                row['Name'] = 'Dummy ' + i.split()[1]
missing_names.isna().sum()
org_indices_with_missing_names = missing_names.index
temp_df = missing_names.reset_index()
print(org_indices_with_missing_names)
j = 0
for i in org_indices_with_missing_names:
    train_df.at[i, 'Name'] = temp_df.iloc[j]['Name']
    j += 1
train_df['Name'] = train_df['Name'].fillna(method='ffill')
test_df['Name'] = test_df['Name'].fillna(method='ffill')
train_df['Name'].isnull().sum()
test_df['Name'].isnull().sum()
train_df[['FirstName', 'LastName']] = train_df['Name'].str.split(' ', expand=True)
test_df[['FirstName', 'LastName']] = test_df['Name'].str.split(' ', expand=True)
train_df['LastName'].value_counts()
train_df.loc[train_df['LastName'] == 'Acobson']
train_df.groupby(by=['LastName'])['Destination'].unique()
train_df.groupby(by=['LastName'])['HomePlanet'].nunique()
train_relatives = train_df.groupby('LastName')['PassengerId'].count().reset_index()
train_relatives = train_relatives.rename(columns={'PassengerId': 'NumRelatives'})
train_relatives
train_df = train_df.merge(train_relatives[['LastName', 'NumRelatives']], how='left', on=['LastName'])
train_df.head()
test_relatives = test_df.groupby('LastName')['PassengerId'].count().reset_index()
test_relatives = test_relatives.rename(columns={'PassengerId': 'NumRelatives'})
test_df = test_df.merge(test_relatives[['LastName', 'NumRelatives']], how='left', on=['LastName'])
train_grpsize = train_df.groupby('Group')['PassengerId'].count().reset_index()
train_grpsize = train_grpsize.rename(columns={'PassengerId': 'GroupSize'})
train_grpsize
train_df = train_df.merge(train_grpsize[['Group', 'GroupSize']], how='left', on=['Group'])
train_df.head()
test_grpsize = test_df.groupby('Group')['PassengerId'].count().reset_index()
test_grpsize = test_grpsize.rename(columns={'PassengerId': 'GroupSize'})
test_df = test_df.merge(test_grpsize[['Group', 'GroupSize']], how='left', on=['Group'])
test_df.head()
train_df['TotalCost'] = train_df['RoomService'] + train_df['FoodCourt'] + train_df['ShoppingMall'] + train_df['Spa'] + train_df['VRDeck']
test_df['TotalCost'] = test_df['RoomService'] + test_df['FoodCourt'] + test_df['ShoppingMall'] + test_df['Spa'] + test_df['VRDeck']
train_df.head()
train_df.groupby(by=['LastName']).sum()['TotalCost'].sort_values()
train_df[train_df['LastName'] == 'Hetforhaft']
print(train_df['HomePlanet'].isnull().sum())
print(test_df['HomePlanet'].isnull().sum())
print(train_df['HomePlanet'].mode()[0])
train_df['HomePlanet'] = train_df['HomePlanet'].fillna(train_df['HomePlanet'].mode()[0])
test_df['HomePlanet'] = test_df['HomePlanet'].fillna(train_df['HomePlanet'].mode()[0])
print(train_df['HomePlanet'].isnull().sum())
print(test_df['HomePlanet'].isnull().sum())
train_df['CryoSleep'].isnull().sum()
train_df['CryoSleep']

def cryosleep_values(row):
    if row['TotalCost'] == 0 and type(row['CryoSleep']) == float:
        return True
    elif type(row['CryoSleep']) == float:
        return False
    else:
        return row['CryoSleep']
train_df['CryoSleep'] = train_df.apply(lambda row: cryosleep_values(row), axis=1)
train_df['CryoSleep'].isnull().sum()
test_df['CryoSleep'].isnull().sum()
test_df['CryoSleep'] = test_df.apply(lambda row: cryosleep_values(row), axis=1)
print(test_df['CryoSleep'].isnull().sum())
print(train_df['Cabin'].isnull().sum())
print(test_df['Cabin'].isnull().sum())
train_df['Cabin'] = train_df['Cabin'].fillna(train_df['Cabin'].mode()[0])
test_df['Cabin'] = test_df['Cabin'].fillna(train_df['Cabin'].mode()[0])
print(train_df['Cabin'].isnull().sum())
print(test_df['Cabin'].isnull().sum())
train_df[['CabinDeck', 'CabinNum', 'CabinSide']] = train_df['Cabin'].str.split('/', expand=True)
test_df[['CabinDeck', 'CabinNum', 'CabinSide']] = test_df['Cabin'].str.split('/', expand=True)
train_df.head(10)
sns.histplot(data=train_df, x='CabinDeck', hue='Transported', multiple='stack', shrink=0.8)
sns.histplot(data=train_df, x='CabinSide', hue='Transported', multiple='stack', shrink=0.8)
print(train_df['VIP'].isnull().sum())
print(test_df['VIP'].isnull().sum())
train_df['VIP'].value_counts()
sns.displot(train_df[train_df['VIP'] == True]['Age'])
plt.title('Age distribution of VIP passengers')
sns.displot(train_df[train_df['VIP'] == True]['TotalCost'])
plt.title('Spend distribution of VIP passengers')
train_df['VIP'] = train_df['VIP'].fillna(False)
test_df['VIP'] = test_df['VIP'].fillna(False)
train_df['VIP'] = train_df.VIP.apply(lambda x: str(x))
test_df['VIP'] = test_df.VIP.apply(lambda x: str(x))
train_df['CryoSleep'] = train_df.CryoSleep.apply(lambda x: str(x))
test_df['CryoSleep'] = test_df.CryoSleep.apply(lambda x: str(x))
train_df.dtypes
train_df1 = train_df.drop(['PassengerId', 'Name', 'PersonNumber', 'CabinNum'], axis=1)
test_df1 = test_df.drop(['PassengerId', 'Name', 'PersonNumber', 'CabinNum'], axis=1)
from sklearn.preprocessing import LabelEncoder

def label_encoding(train, test, columns_train, columns_test):
    train = train.copy()
    test = test.copy()
    for col in columns_train:
        encoder = LabelEncoder()
        train[col] = encoder.fit_transform(train[col])
    for col in columns_test:
        encoder = LabelEncoder()
        test[col] = encoder.fit_transform(test[col])
    return (train, test)
categorical_features_train = train_df1.select_dtypes('object').columns.to_list()
categorical_features_test = test_df1.select_dtypes('object').columns.to_list()
print(categorical_features_train)
print(train_df1.dtypes)
(train_df2, test_df2) = label_encoding(train_df1, test_df1, categorical_features_train, categorical_features_test)
train_df2.head()
test_df2.head()
train_df2 = pd.get_dummies(train_df2, columns=['HomePlanet', 'Destination', 'AgeGroup', 'CabinDeck', 'CabinSide', 'CryoSleep', 'VIP'])
test_df2 = pd.get_dummies(test_df2, columns=['HomePlanet', 'Destination', 'AgeGroup', 'CabinDeck', 'CabinSide', 'CryoSleep', 'VIP'])
train_df2
from sklearn.model_selection import train_test_split
X = train_df2.drop(['Transported'], axis=1)
y = train_df2['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
random_forest_clf = RandomForestClassifier(max_depth=10, random_state=101)