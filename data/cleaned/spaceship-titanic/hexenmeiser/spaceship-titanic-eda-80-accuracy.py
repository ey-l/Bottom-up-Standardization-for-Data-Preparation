import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
data.head()
data.shape
data['TotalSpend'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['VRDeck']
data[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = data['Cabin'].str.split('/', expand=True)
data['Cabin_Deck_Side'] = data['Cabin_Deck'] + '/' + data['Cabin_Side']
data['pass_group'] = data['PassengerId'].apply(lambda x: x.split('_')[0])
data['pass_group_num'] = data['PassengerId'].apply(lambda x: x.split('_')[1])
data[['FirstName', 'LastName']] = data['Name'].str.split(' ', expand=True)
family_count = pd.DataFrame(data[['pass_group', 'LastName']].value_counts())
family_count.rename(columns={0: 'FamilyCount'}, inplace=True)
data = data.join(family_count, on=['pass_group', 'LastName'], rsuffix='fm')
data['FamilyFlag'] = data['FamilyCount'] > 1
group_size = data['pass_group'].value_counts()
data = data.join(group_size, on='pass_group', rsuffix='test')
data.rename(columns={'pass_grouptest': 'groupsize'}, inplace=True)
Home_planets = data[['pass_group', 'HomePlanet']]
Home_planets.dropna(inplace=True)
Home_planets.drop_duplicates(inplace=True)
Home_planets
Home_planets.set_index('pass_group', inplace=True)
data = data.join(Home_planets, on='pass_group', rsuffix='_group')
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 5, 12, 18, 30, 60, np.inf], labels=['Infant', 'Child', 'Teen', 'Adult', 'Elderly', 'Senior'])
data['Alone'] = data['groupsize'] == 1
data.info()
data.isnull().sum()
sns.heatmap(data.isnull(), cbar=False)
print('Number of rows with nulls: {}'.format(data.shape[0] - data.dropna().shape[0]))
print('% of data lost if all null rows were dropped: {:2.2%}'.format(1 - data.dropna().shape[0] / data.shape[0]))
data.describe()
data.nunique()
data.corr()
sns.heatmap(data.corr())
data['HomePlanet'].value_counts().plot(kind='bar')
sns.histplot(data, x='HomePlanet', hue='Transported')
data['CryoSleep'].value_counts().plot(kind='bar')
sns.histplot(data, x=data['CryoSleep'].dropna().astype('int'), hue='Transported')
data['Destination'].value_counts().plot(kind='bar')
sns.histplot(data, x='Destination', hue='Transported')
data['VIP'].value_counts().plot(kind='bar')
sns.histplot(data, x=data['VIP'].dropna().astype(int), hue='Transported')
data['Transported'].value_counts().plot(kind='bar')
data['Transported'].value_counts()
data['Cabin_Deck'].value_counts().plot(kind='bar')
sns.histplot(data, x=data['Cabin_Deck'].dropna(), hue='Transported')
data['Cabin_Number'].nunique()
data['Cabin_Side'].value_counts()
sns.histplot(data, x=data['Cabin_Side'], hue='Transported')
(fig, ax) = plt.subplots()
ax.hist(data[data['Transported'] == 1]['Age'], bins=80, alpha=0.5, color='blue', label='Transported')
ax.hist(data[data['Transported'] == 0]['Age'], bins=80, alpha=0.5, color='red', label='Not Transported')
ax.set_xlabel('Age')
ax.set_ylabel('Count of passengers')
fig.suptitle('Age vs. Transported for Spaceship Titanic Passengers')
ax.legend()
sns.histplot(data, x=data['AgeGroup'], hue='Transported')
sns.boxplot(data=data, x='Transported', y='RoomService')
plt.hist(data['TotalSpend'], bins=45, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
bins = [0, 250, 500, 750, 1000, 1500, 2500, np.inf]
labels = ["'0-250'", "'250-500'", "'500-750'", "'750-1000'", "'1000-1500'", "'1500-2500'", "'2500+'"]
data['TotalSpendBuckets'] = pd.cut(data['TotalSpend'], bins=bins, labels=labels)
sns.histplot(data, x=data['TotalSpendBuckets'].sort_values(), hue='Transported')
bins = [0, 25, 50, 75, 100, 150, 250, 500, 1000, np.inf]
labels = ["'0-25'", "'25-50'", "'50-75'", "'75-100'", "'100-150'", "'150-250'", "'250-500'", "'500-1000'", "'1,000+'"]
sns.histplot(data, x=pd.cut(data['RoomService'], bins=bins, labels=labels).sort_values(), hue='Transported')
bins = [0, 25, 50, 75, 100, 150, 250, 500, 1000, np.inf]
labels = ["'0-25'", "'25-50'", "'50-75'", "'75-100'", "'100-150'", "'150-250'", "'250-500'", "'500-1000'", "'1,000+'"]
sns.histplot(data, x=pd.cut(data['FoodCourt'], bins=bins, labels=labels).sort_values(), hue='Transported')
bins = [0, 25, 50, 75, 100, 150, 250, 500, 1000, np.inf]
labels = ["'0-25'", "'25-50'", "'50-75'", "'75-100'", "'100-150'", "'150-250'", "'250-500'", "'500-1000'", "'1,000+'"]
sns.histplot(data, x=pd.cut(data['ShoppingMall'], bins=bins, labels=labels).sort_values(), hue='Transported')
bins = [0, 25, 50, 75, 100, 150, 250, 500, 1000, np.inf]
labels = ["'0-25'", "'25-50'", "'50-75'", "'75-100'", "'100-150'", "'150-250'", "'250-500'", "'500-1000'", "'1,000+'"]
sns.histplot(data, x=pd.cut(data['Spa'], bins=bins, labels=labels).sort_values(), hue='Transported')
bins = [0, 25, 50, 75, 100, 150, 250, 500, 1000, np.inf]
labels = ["'0-25'", "'25-50'", "'50-75'", "'75-100'", "'100-150'", "'150-250'", "'250-500'", "'500-1000'", "'1,000+'"]
sns.histplot(data, x=pd.cut(data['VRDeck'], bins=bins, labels=labels).sort_values(), hue='Transported')
data.groupby(by=['CryoSleep']).sum()
data.CryoSleep[data['TotalSpend'] == 0].value_counts()
sns.histplot(data, x=data['FamilyCount'].dropna().astype('int'), hue='Transported')
sns.histplot(data, x=data['FamilyFlag'].dropna().astype('int'), hue='Transported')
sns.histplot(data, x=data['groupsize'].dropna().astype('int'), hue='Transported')
sns.histplot(data, x=data['Alone'].dropna().astype('int'), hue='Transported')
import math
for i in range(0, len(data)):
    if data.loc[i, 'TotalSpend'] > 0 and math.isnan(data.loc[i, 'CryoSleep']) == True:
        data.loc[i, 'CryoSleep'] = False
    elif data.loc[i, 'TotalSpend'] == 0 and math.isnan(data.loc[i, 'CryoSleep']) == True:
        data.loc[i, 'CryoSleep'] = True
from sklearn.model_selection import train_test_split
X = data[['HomePlanet_group', 'CryoSleep', 'Cabin_Deck_Side', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'FamilyCount', 'groupsize']]
y = data['Transported']
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