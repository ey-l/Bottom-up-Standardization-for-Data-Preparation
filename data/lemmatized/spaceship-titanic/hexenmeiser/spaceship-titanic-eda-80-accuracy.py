import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.shape
_input1['TotalSpend'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['VRDeck']
_input1[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = _input1['Cabin'].str.split('/', expand=True)
_input1['Cabin_Deck_Side'] = _input1['Cabin_Deck'] + '/' + _input1['Cabin_Side']
_input1['pass_group'] = _input1['PassengerId'].apply(lambda x: x.split('_')[0])
_input1['pass_group_num'] = _input1['PassengerId'].apply(lambda x: x.split('_')[1])
_input1[['FirstName', 'LastName']] = _input1['Name'].str.split(' ', expand=True)
family_count = pd.DataFrame(_input1[['pass_group', 'LastName']].value_counts())
family_count = family_count.rename(columns={0: 'FamilyCount'}, inplace=False)
_input1 = _input1.join(family_count, on=['pass_group', 'LastName'], rsuffix='fm')
_input1['FamilyFlag'] = _input1['FamilyCount'] > 1
group_size = _input1['pass_group'].value_counts()
_input1 = _input1.join(group_size, on='pass_group', rsuffix='test')
_input1 = _input1.rename(columns={'pass_grouptest': 'groupsize'}, inplace=False)
Home_planets = _input1[['pass_group', 'HomePlanet']]
Home_planets = Home_planets.dropna(inplace=False)
Home_planets = Home_planets.drop_duplicates(inplace=False)
Home_planets
Home_planets = Home_planets.set_index('pass_group', inplace=False)
_input1 = _input1.join(Home_planets, on='pass_group', rsuffix='_group')
_input1['AgeGroup'] = pd.cut(_input1['Age'], bins=[0, 5, 12, 18, 30, 60, np.inf], labels=['Infant', 'Child', 'Teen', 'Adult', 'Elderly', 'Senior'])
_input1['Alone'] = _input1['groupsize'] == 1
_input1.info()
_input1.isnull().sum()
sns.heatmap(_input1.isnull(), cbar=False)
print('Number of rows with nulls: {}'.format(_input1.shape[0] - _input1.dropna().shape[0]))
print('% of data lost if all null rows were dropped: {:2.2%}'.format(1 - _input1.dropna().shape[0] / _input1.shape[0]))
_input1.describe()
_input1.nunique()
_input1.corr()
sns.heatmap(_input1.corr())
_input1['HomePlanet'].value_counts().plot(kind='bar')
sns.histplot(_input1, x='HomePlanet', hue='Transported')
_input1['CryoSleep'].value_counts().plot(kind='bar')
sns.histplot(_input1, x=_input1['CryoSleep'].dropna().astype('int'), hue='Transported')
_input1['Destination'].value_counts().plot(kind='bar')
sns.histplot(_input1, x='Destination', hue='Transported')
_input1['VIP'].value_counts().plot(kind='bar')
sns.histplot(_input1, x=_input1['VIP'].dropna().astype(int), hue='Transported')
_input1['Transported'].value_counts().plot(kind='bar')
_input1['Transported'].value_counts()
_input1['Cabin_Deck'].value_counts().plot(kind='bar')
sns.histplot(_input1, x=_input1['Cabin_Deck'].dropna(), hue='Transported')
_input1['Cabin_Number'].nunique()
_input1['Cabin_Side'].value_counts()
sns.histplot(_input1, x=_input1['Cabin_Side'], hue='Transported')
(fig, ax) = plt.subplots()
ax.hist(_input1[_input1['Transported'] == 1]['Age'], bins=80, alpha=0.5, color='blue', label='Transported')
ax.hist(_input1[_input1['Transported'] == 0]['Age'], bins=80, alpha=0.5, color='red', label='Not Transported')
ax.set_xlabel('Age')
ax.set_ylabel('Count of passengers')
fig.suptitle('Age vs. Transported for Spaceship Titanic Passengers')
ax.legend()
sns.histplot(_input1, x=_input1['AgeGroup'], hue='Transported')
sns.boxplot(data=_input1, x='Transported', y='RoomService')
plt.hist(_input1['TotalSpend'], bins=45, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
bins = [0, 250, 500, 750, 1000, 1500, 2500, np.inf]
labels = ["'0-250'", "'250-500'", "'500-750'", "'750-1000'", "'1000-1500'", "'1500-2500'", "'2500+'"]
_input1['TotalSpendBuckets'] = pd.cut(_input1['TotalSpend'], bins=bins, labels=labels)
sns.histplot(_input1, x=_input1['TotalSpendBuckets'].sort_values(), hue='Transported')
bins = [0, 25, 50, 75, 100, 150, 250, 500, 1000, np.inf]
labels = ["'0-25'", "'25-50'", "'50-75'", "'75-100'", "'100-150'", "'150-250'", "'250-500'", "'500-1000'", "'1,000+'"]
sns.histplot(_input1, x=pd.cut(_input1['RoomService'], bins=bins, labels=labels).sort_values(), hue='Transported')
bins = [0, 25, 50, 75, 100, 150, 250, 500, 1000, np.inf]
labels = ["'0-25'", "'25-50'", "'50-75'", "'75-100'", "'100-150'", "'150-250'", "'250-500'", "'500-1000'", "'1,000+'"]
sns.histplot(_input1, x=pd.cut(_input1['FoodCourt'], bins=bins, labels=labels).sort_values(), hue='Transported')
bins = [0, 25, 50, 75, 100, 150, 250, 500, 1000, np.inf]
labels = ["'0-25'", "'25-50'", "'50-75'", "'75-100'", "'100-150'", "'150-250'", "'250-500'", "'500-1000'", "'1,000+'"]
sns.histplot(_input1, x=pd.cut(_input1['ShoppingMall'], bins=bins, labels=labels).sort_values(), hue='Transported')
bins = [0, 25, 50, 75, 100, 150, 250, 500, 1000, np.inf]
labels = ["'0-25'", "'25-50'", "'50-75'", "'75-100'", "'100-150'", "'150-250'", "'250-500'", "'500-1000'", "'1,000+'"]
sns.histplot(_input1, x=pd.cut(_input1['Spa'], bins=bins, labels=labels).sort_values(), hue='Transported')
bins = [0, 25, 50, 75, 100, 150, 250, 500, 1000, np.inf]
labels = ["'0-25'", "'25-50'", "'50-75'", "'75-100'", "'100-150'", "'150-250'", "'250-500'", "'500-1000'", "'1,000+'"]
sns.histplot(_input1, x=pd.cut(_input1['VRDeck'], bins=bins, labels=labels).sort_values(), hue='Transported')
_input1.groupby(by=['CryoSleep']).sum()
_input1.CryoSleep[_input1['TotalSpend'] == 0].value_counts()
sns.histplot(_input1, x=_input1['FamilyCount'].dropna().astype('int'), hue='Transported')
sns.histplot(_input1, x=_input1['FamilyFlag'].dropna().astype('int'), hue='Transported')
sns.histplot(_input1, x=_input1['groupsize'].dropna().astype('int'), hue='Transported')
sns.histplot(_input1, x=_input1['Alone'].dropna().astype('int'), hue='Transported')
import math
for i in range(0, len(_input1)):
    if _input1.loc[i, 'TotalSpend'] > 0 and math.isnan(_input1.loc[i, 'CryoSleep']) == True:
        _input1.loc[i, 'CryoSleep'] = False
    elif _input1.loc[i, 'TotalSpend'] == 0 and math.isnan(_input1.loc[i, 'CryoSleep']) == True:
        _input1.loc[i, 'CryoSleep'] = True
from sklearn.model_selection import train_test_split
X = _input1[['HomePlanet_group', 'CryoSleep', 'Cabin_Deck_Side', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'FamilyCount', 'groupsize']]
y = _input1['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=40)
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'FamilyCount', 'groupsize']
bin_features = ['CryoSleep', 'VIP']
cat_features = ['HomePlanet_group', 'Cabin_Deck_Side', 'Destination']
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
num_pipeline = Pipeline([('std_scaler', StandardScaler()), ('knn_num_imputer', KNNImputer(n_neighbors=10))])
bin_pipeline = Pipeline([('knn_bin_imputer', KNNImputer(n_neighbors=10))])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder())])
from sklearn.compose import ColumnTransformer
full_pipeline = ColumnTransformer([('num', num_pipeline, num_features), ('bin', bin_pipeline, bin_features), ('cat', cat_pipeline, cat_features)])
X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
naive_b = GaussianNB()