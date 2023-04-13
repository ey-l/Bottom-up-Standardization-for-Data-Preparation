import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('data/input/spaceship-titanic/train.csv').copy()
test = pd.read_csv('data/input/spaceship-titanic/test.csv').copy()
train.head()
test.head()
train.shape
train.isnull().sum()
test.isnull().sum()
df = pd.concat([train, test], ignore_index=True)
df
df.tail()
df.info()
df.describe()
df['HomePlanet'].unique()
df['HomePlanet'].dropna().value_counts()
sns.barplot(x=df['HomePlanet'].dropna().value_counts().keys(), y=df['HomePlanet'].dropna().value_counts().values)
df['CryoSleep'].unique()
df['CryoSleep'].value_counts()
sns.barplot(x=df['CryoSleep'].value_counts().keys(), y=df['CryoSleep'].value_counts().values)
df['Cabin'].value_counts()
df['Destination'].unique()
df['Destination'].value_counts()
sns.barplot(x=df['Destination'].value_counts().keys(), y=df['Destination'].value_counts().values)
df['Age']
sns.displot(df['Age'].dropna())
sns.boxplot(df['Age'].dropna())
df['Age'].dropna().mean()
df['VIP'].unique()
sns.barplot(x=df['VIP'].dropna().value_counts().keys(), y=df['VIP'].dropna().value_counts().values)
df['RoomService']
total = [df['RoomService'].mean(), df['FoodCourt'].mean(), df['ShoppingMall'].mean(), df['Spa'].mean(), df['VRDeck'].mean()]
labels = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
total
plt.pie(total, labels=labels, autopct='%.1f%%')
for i in labels:
    sns.distplot(df[i].dropna())
for i in labels:
    sns.boxplot(df[i].dropna())
df.isnull().sum()
for i in labels:
    df[i] = df[i].fillna(df[i].median(), inplace=False)
df['total_expenditure'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
df.head()
for i in labels:
    df = df.drop(i, axis=1, inplace=False)
df['Age'] = df['Age'].fillna(df['Age'].median(), inplace=False)
df = df.drop('Name', axis=1, inplace=False)
df
df['VIP']
df['VIP'].unique()

def f(string):
    if string == False:
        return 0
    elif string == True:
        return 1
    else:
        return 0
df['CryoSleep'] = df['CryoSleep'].apply(f)
df['CryoSleep'].value_counts()
df['VIP'] = df['VIP'].apply(f)
df.head()
df['Destination'] = df['Destination'].fillna('TRAPPIST-1e', inplace=False)
df['Destination'].value_counts()
df1 = pd.get_dummies(df['Destination'])
df['HomePlanet'].value_counts()
df['HomePlanet'] = df['HomePlanet'].fillna('Earth', inplace=False)
df2 = pd.get_dummies(df['HomePlanet'])
df2
df = pd.concat([df, df1, df2], axis='columns')
df
df = df.drop('HomePlanet', axis=1, inplace=False)
df = df.drop('Destination', axis=1, inplace=False)
df
df['Cabin'] = df['Cabin'].str.split('/').str[2]
df['Cabin'].value_counts()
df['Cabin'] = df['Cabin'].fillna('M', inplace=False)
df3 = pd.get_dummies(df['Cabin'])
df = pd.concat([df, df3], axis='columns')
df = df.drop('Cabin', axis=1, inplace=False)
df
final_train = df[0:8693]
final_test = df[8693:]
final_train
final_test
final_test = final_test.drop('Transported', axis=1, inplace=False)
final_test = final_test.drop('PassengerId', axis=1, inplace=False)
final_train = final_train.drop('PassengerId', axis=1, inplace=False)
final_train['Transported'] = final_train['Transported'].apply(f)
Y_train = final_train['Transported']
Y_train
final_train = final_train.drop('Transported', axis=1, inplace=False)
X_train = final_train
X_train
Y_train
X_test = final_test
X_test
X_train
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()