import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
submission
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train
train.isnull().sum()
train1 = train.dropna()
train1
train1['Transported'].replace({False: 0, True: 1}, inplace=True)
train1['Transported']
sns.displot(train1['Transported'])
trans_count = train1['Transported'].value_counts()
trans_count
trans_percent = trans_count / len(train)
trans_percent
plt.figure(figsize=(25, 7))
ax = plt.subplot()
ax.scatter(train1[train1['Transported'] == 1]['Age'], train1[train['Transported'] == 1]['FoodCourt'], c='green', s=train1[train1['Transported'] == 1]['VRDeck'])
ax.scatter(train1[train1['Transported'] == 0]['Age'], train1[train['Transported'] == 0]['FoodCourt'], c='red', s=train1[train1['Transported'] == 0]['VRDeck'])
target = train['Transported']
train1.drop(['Transported'], axis=1, inplace=True)
train1
combi = train1.append(test)
combi
combi.info()
combi.describe()
combi.isnull().sum()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(random_state=42)
data = pd.Timestamp('2200-01-01')
for col in combi:
    if combi[col].dtype == 'object':
        combi[col].fillna('not listed ', inplace=True)
    if combi[col].dtype == 'int':
        combi[col].fillna(combi[col].mean(), inplace=True)
    if combi[col].dtype == 'float':
        combi[col] = imp.fit_transform(combi[col].values.reshape(-1, 1))
    if combi[col].dtype == 'datetime64[ns]':
        combi[col].fillna(data, inplace=True)
combi
combi.isnull().sum()
sns.displot(combi['HomePlanet'])
home_count = combi['HomePlanet'].value_counts()
home_count
home_percent = home_count / len(combi)
home_percent
mylabels = ['Earth', 'Europa', 'Mars', 'not listed']
plt.pie(home_percent, labels=mylabels)

combi['HomePlanet'].replace({'Earth': 1, 'Europa': 2, 'Mars': 3, 'not listed': 4}, inplace=True)
combi['HomePlanet']
print(combi.iloc[6674])
combi['CryoSleep'].replace({False: 0, True: 1})
sleep_count = combi['CryoSleep'].value_counts()
sleep_count
sleep_percent = sleep_count / len(combi)
sleep_percent
sns.displot(combi['Destination'])
dest_count = combi['Destination'].value_counts()
dest_count
dest_percent = dest_count / len(combi)
dest_percent
mylabels = ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22', 'not listed']
plt.pie(dest_percent, labels=mylabels)

combi['Destination'].replace({'TRAPPIST-1e': 1, '55 Cancri e': 2, 'PSO J318.5-22': 3, 'not listed': 4}, inplace=True)
combi['Destination']
combi['Age_group'] = pd.cut(x=combi['Age'], bins=[-1, 18, 40, 65, 100], labels=['child', 'young adult', 'middle age', 'pensioner'])
combi['Age_group']
sns.displot(combi['Age_group'])
age_count = combi['Age_group'].value_counts()
age_count
age_percent = age_count / len(combi)
age_percent
mylabels = ['young adult', 'child', 'middle age', 'pensioner']
plt.pie(age_percent, labels=mylabels)

combi['Age_group'].replace({'young adult': 1, 'child': 2, 'middle age': 3, 'pensioner': 4}, inplace=True)
combi['Age_group']
combi['Age_group'] = combi['Age_group'].astype(int)
combi['Age_group']
combi['VIP'].replace({False: 0, True: 1})
vip_count = combi['VIP'].value_counts()
vip_count
vip_percent = vip_count / len(combi)
vip_percent
sns.violinplot(combi['RoomService'])
rm_service_high = combi['RoomService'].max()
print(rm_service_high)
combi['Room_Service_group'] = pd.cut(x=combi['RoomService'], bins=[-1, 2000, 8000, 12000], labels=['low', 'mid', 'high'])
combi['Room_Service_group']
sns.displot(combi['Room_Service_group'])
rm_service_count = combi['Room_Service_group'].value_counts()
rm_service_count
rm_service_percent = rm_service_count / len(combi)
rm_service_percent
mylabels = ['low', 'mid', 'high']
plt.pie(rm_service_percent, labels=mylabels)

combi['Room_Service_group'].replace({'low': 1, 'mid': 2, 'high': 3}, inplace=True)
combi['Room_Service_group']
sns.violinplot(combi['Spa'])
spa_high = combi['Spa'].max()
print(spa_high)
combi['Spa_group'] = pd.cut(x=combi['Spa'], bins=[-1, 5000, 15000, 23000], labels=['low', 'mid', 'high'])
combi['Spa_group']
sns.displot(combi['Spa_group'])
spa_count = combi['Spa_group'].value_counts()
spa_count
sns.violinplot(combi['VRDeck'])
vr_high = combi['VRDeck'].max()
print(vr_high)
combi['VR_group'] = pd.cut(x=combi['VRDeck'], bins=[-1, 5000, 15000, 23000], labels=['low', 'mid', 'high'])
combi['VR_group']
combi.info()