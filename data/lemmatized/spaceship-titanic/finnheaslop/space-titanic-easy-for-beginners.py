import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head(100)
_input1.info()
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
_input1['CryoSleep'] = l1.fit_transform(_input1['CryoSleep'])
_input1['Cabin'] = l1.fit_transform(_input1['Cabin'])
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].mean(), inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna(_input1['Cabin'].mean(), inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].mean(), inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].mean(), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].mean(), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].mean(), inplace=False)
_input1.info()
_input1.describe()
_input1.head(5)
sns.countplot(x='HomePlanet', data=_input1)
sns.catplot(x='HomePlanet', kind='count', hue='Transported', data=_input1)
_input1.CryoSleep.value_counts().plot(kind='pie', figsize=(12, 5), autopct='%0.1f%%')
plt.xlabel('Percentage of Passengers CryoSleeping')
plt.ylabel('')
sns.catplot(x='CryoSleep', kind='count', hue='Transported', data=_input1)
_input1.pivot_table(index='CryoSleep', columns='Transported', aggfunc={'Transported': 'count'})
sns.countplot(x='Destination', data=_input1)
df_count = _input1[['Age']].apply(pd.value_counts)
df_count.plot(kind='bar', color='Grey', figsize=(12, 12))
plt.xticks(rotation=90)
plt.title('Most Common Ages')
healthy = _input1[_input1['Age'] <= 80]
age_s = sns.catplot(x='Age', kind='count', hue='Transported', height=20, aspect=11.7 / 8.27, data=healthy)
plt.legend(title_fontsize='400')
_input1.head(3)
(fig, axes) = plt.subplots(2, 3, sharey=True, figsize=(18, 10))
_input1.plot.scatter(x='RoomService', y='Age', ax=axes[0, 0], color='Purple')
_input1.plot.scatter(x='FoodCourt', y='Age', ax=axes[0, 1], color='Red')
_input1.plot.scatter(x='ShoppingMall', y='Age', ax=axes[0, 2], color='Blue')
_input1.plot.scatter(x='Spa', y='Age', ax=axes[1, 0], color='Black')
_input1.plot.scatter(x='VRDeck', y='Age', ax=axes[1, 1], color='Orange')