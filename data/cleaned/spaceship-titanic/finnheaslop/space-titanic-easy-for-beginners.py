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
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head(100)
df.info()
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
df['CryoSleep'] = l1.fit_transform(df['CryoSleep'])
df['Cabin'] = l1.fit_transform(df['Cabin'])
df['CryoSleep'].fillna(df['CryoSleep'].mean(), inplace=True)
df['Cabin'].fillna(df['Cabin'].mean(), inplace=True)
df['VIP'].fillna(df['VIP'].mean(), inplace=True)
df['RoomService'].fillna(df['RoomService'].mean(), inplace=True)
df['Spa'].fillna(df['Spa'].mean(), inplace=True)
df['VRDeck'].fillna(df['VRDeck'].mean(), inplace=True)
df.info()
df.describe()
df.head(5)
sns.countplot(x='HomePlanet', data=df)
sns.catplot(x='HomePlanet', kind='count', hue='Transported', data=df)

df.CryoSleep.value_counts().plot(kind='pie', figsize=(12, 5), autopct='%0.1f%%')
plt.xlabel('Percentage of Passengers CryoSleeping')
plt.ylabel('')

sns.catplot(x='CryoSleep', kind='count', hue='Transported', data=df)
df.pivot_table(index='CryoSleep', columns='Transported', aggfunc={'Transported': 'count'})
sns.countplot(x='Destination', data=df)
df_count = df[['Age']].apply(pd.value_counts)
df_count.plot(kind='bar', color='Grey', figsize=(12, 12))
plt.xticks(rotation=90)
plt.title('Most Common Ages')

healthy = df[df['Age'] <= 80]
age_s = sns.catplot(x='Age', kind='count', hue='Transported', height=20, aspect=11.7 / 8.27, data=healthy)
plt.legend(title_fontsize='400')

df.head(3)
(fig, axes) = plt.subplots(2, 3, sharey=True, figsize=(18, 10))
df.plot.scatter(x='RoomService', y='Age', ax=axes[0, 0], color='Purple')
df.plot.scatter(x='FoodCourt', y='Age', ax=axes[0, 1], color='Red')
df.plot.scatter(x='ShoppingMall', y='Age', ax=axes[0, 2], color='Blue')
df.plot.scatter(x='Spa', y='Age', ax=axes[1, 0], color='Black')
df.plot.scatter(x='VRDeck', y='Age', ax=axes[1, 1], color='Orange')
