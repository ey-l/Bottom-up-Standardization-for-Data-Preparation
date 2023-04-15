import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df.describe()
import matplotlib.pyplot as plt
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, annot=True)

df['FoodCourt_cat'] = pd.qcut(df['FoodCourt'], 20, duplicates='drop')
test['FoodCourt_cat'] = pd.qcut(test['FoodCourt'], 20, duplicates='drop')
df['Spa_cat'] = pd.qcut(df['Spa'], 20, duplicates='drop')
test['Spa_cat'] = pd.qcut(test['Spa'], 20, duplicates='drop')
df['VRDeck_cat'] = pd.qcut(df['VRDeck'], 20, duplicates='drop')
test['VRDeck_cat'] = pd.qcut(test['VRDeck'], 20, duplicates='drop')
cat_cols = ['FoodCourt_cat', 'Spa_cat', 'VRDeck_cat']
num_cols = ['FoodCourt', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=len(cat_cols), ncols=len(num_cols), figsize=(40, 10))
for (idx1, row) in enumerate(cat_cols):
    for (idx2, col) in enumerate(num_cols):
        sns.barplot(x=row, y=col, data=df, ax=axs[idx1][idx2])
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
(fig, axs) = plt.subplots(nrows=len(cat_cols), ncols=len(num_cols), figsize=(30, 20))
for (idx1, row) in enumerate(cat_cols):
    for (idx2, col) in enumerate(num_cols):
        sns.barplot(x=row, y=col, data=df, ax=axs[idx1][idx2])
print(df.isnull().sum())
print(test.isnull().sum())
df['Expenditure'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
test['Expenditure'] = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']
df.loc[df['RoomService'] > 0, 'CryoSleep'] = df['CryoSleep'].fillna(False)
test.loc[test['RoomService'] > 0, 'CryoSleep'] = test['CryoSleep'].fillna(False)
df.loc[df['FoodCourt'] > 0, 'CryoSleep'] = df['CryoSleep'].fillna(False)
test.loc[test['FoodCourt'] > 0, 'CryoSleep'] = test['CryoSleep'].fillna(False)
df.loc[df['ShoppingMall'] > 0, 'CryoSleep'] = df['CryoSleep'].fillna(False)
test.loc[test['ShoppingMall'] > 0, 'CryoSleep'] = test['CryoSleep'].fillna(False)
df.loc[df['Spa'] > 0, 'CryoSleep'] = df['CryoSleep'].fillna(False)
test.loc[test['Spa'] > 0, 'CryoSleep'] = test['CryoSleep'].fillna(False)
df.loc[df['VRDeck'] > 0, 'CryoSleep'] = df['CryoSleep'].fillna(False)
test.loc[test['VRDeck'] > 0, 'CryoSleep'] = test['CryoSleep'].fillna(False)
df.loc[df['Expenditure'] == 0, 'CryoSleep'] = df['CryoSleep'].fillna(True)
test.loc[test['Expenditure'] == 0, 'CryoSleep'] = test['CryoSleep'].fillna(True)
print(df.isnull().sum())
print(test.isnull().sum())
df[df['CryoSleep'].isnull()]
df['CryoSleep'] = df['CryoSleep'].fillna(True)
test['CryoSleep'] = test['CryoSleep'].fillna(True)
print(df.isnull().sum())
print(test.isnull().sum())
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(30, 4))
for (idx, col) in enumerate(num_cols):
    sns.scatterplot(x='Transported', y=col, data=df, ax=axs[idx])
(fig, axs) = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(30, 3))
for (idx, col) in enumerate(num_cols):
    sns.histplot(df[col], bins=100, ax=axs[idx])

outlier_r = df[df['RoomService'] > 8000].index
df = df.drop(outlier_r, axis=0)
outlier_f = df[df['FoodCourt'] > 20000].index
df = df.drop(outlier_f, axis=0)
outlier_sm = df[df['ShoppingMall'] > 9000].index
df = df.drop(outlier_sm, axis=0)
outlier_s = df[df['Spa'] > 12000].index
df = df.drop(outlier_s, axis=0)
outlier_v = df[df['VRDeck'] > 14000].index
df = df.drop(outlier_v, axis=0)
(fig, axs) = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(30, 4))
for (idx, col) in enumerate(num_cols):
    sns.scatterplot(x='Transported', y=col, data=df, ax=axs[idx])
(fig, axs) = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(30, 3))
for (idx, col) in enumerate(num_cols):
    sns.histplot(df[col], bins=100, ax=axs[idx])

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=len(num_cols), ncols=len(num_cols), figsize=(30, 20))
for (idx1, row) in enumerate(num_cols):
    for (idx2, col) in enumerate(num_cols):
        sns.scatterplot(x=row, y=col, data=df, ax=axs[idx1][idx2])

def age_category(age):
    cat = ''
    if age <= 12:
        cat = '0~12'
    elif age <= 17:
        cat = '13~17'
    elif age <= 25:
        cat = '17~25'
    elif age <= 30:
        cat = '26~30'
    elif age <= 50:
        cat = '31~50'
    else:
        cat = '51~'
    return cat
df['Age_cat'] = df['Age'].apply(lambda x: age_category(x))
test['Age_cat'] = test['Age'].apply(lambda x: age_category(x))
cat_cols = ['Age_cat']
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=len(cat_cols), ncols=len(num_cols), figsize=(23, 3))
for (idx, col) in enumerate(num_cols):
    sns.barplot(x='Age_cat', y=col, data=df, ax=axs[idx])
df.isnull().sum()
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in num_cols:
    df.loc[df['CryoSleep'] == True, col] = df[col].fillna(0)
    test.loc[test['CryoSleep'] == True, col] = test[col].fillna(0)
for col in num_cols:
    df.loc[df['Age'] <= 12, col] = df[col].fillna(0)
    test.loc[test['Age'] <= 12, col] = test[col].fillna(0)
df['FoodCourt'] = df['FoodCourt'].fillna(df.groupby('VRDeck_cat')['FoodCourt'].transform('mean'))
test['FoodCourt'] = test['FoodCourt'].fillna(test.groupby('VRDeck_cat')['FoodCourt'].transform('mean'))
df['Spa'] = df['Spa'].fillna(df.groupby('Transported')['Spa'].transform('mean'))
test['Spa'] = test['Spa'].fillna(test.groupby('FoodCourt_cat')['Spa'].transform('mean'))
df['VRDeck'] = df['VRDeck'].fillna(df.groupby('Transported')['VRDeck'].transform('mean'))
test['VRDeck'] = test['VRDeck'].fillna(test.groupby('FoodCourt_cat')['VRDeck'].transform('mean'))
df['RoomService'] = df['RoomService'].fillna(df.groupby('Transported')['RoomService'].transform('mean'))
test['RoomService'] = test['RoomService'].fillna(test['RoomService'].mean())
df['ShoppingMall'] = df['ShoppingMall'].fillna(df['ShoppingMall'].mean())
test['ShoppingMall'] = test['ShoppingMall'].fillna(test['ShoppingMall'].mean())


df['FoodCourt'] = df['FoodCourt'].fillna(df['FoodCourt'].mean())
test['FoodCourt'] = test['FoodCourt'].fillna(test['FoodCourt'].mean())
df['Spa'] = df['Spa'].fillna(df['Spa'].mean())
test['Spa'] = test['Spa'].fillna(test['Spa'].mean())
df['VRDeck'] = df['VRDeck'].fillna(df['VRDeck'].mean())
test['VRDeck'] = test['VRDeck'].fillna(test['VRDeck'].mean())


df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
test['Group'] = test['PassengerId'].apply(lambda x: x.split('_')[0])
import matplotlib.pyplot as plt
import seaborn as sns
GHP_gb = df.groupby(['Group', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)



sns.countplot((GHP_gb > 0).sum(axis=1))

def cabin_cat(c):
    if c[0] == 'A':
        c = 'A'
    elif c[0] == 'B':
        c = 'B'
    elif c[0] == 'C':
        c = 'C'
    elif c[0] == 'D':
        c = 'D'
    elif c[0] == 'E':
        c = 'E'
    elif c[0] == 'F':
        c = 'F'
    elif c[0] == 'G':
        c = 'G'
    else:
        c = 'T'
    return c
cabin_train = df[df['Cabin'].notnull()]
cabin_test = test[test['Cabin'].notnull()]
df['Cabin_cat'] = cabin_train['Cabin'].apply(lambda x: cabin_cat(x))
test['Cabin_cat'] = cabin_test['Cabin'].apply(lambda x: cabin_cat(x))
df.groupby('Cabin_cat')[num_cols].mean()
(fig, axs) = plt.subplots(nrows=len(cat_cols), ncols=len(num_cols), figsize=(30, 4))
for (idx, col) in enumerate(num_cols):
    sns.barplot(x='Cabin_cat', y=col, data=df, ax=axs[idx])
sns.barplot(x='Cabin_cat', y='Transported', data=df)
pd.crosstab(index=df['HomePlanet'], columns=df['Cabin_cat'])
df.loc[df['HomePlanet'].isnull() & df['Cabin_cat'].isin(['A', 'B', 'C', 'T']), 'HomePlanet'] = 'Europa'
df.loc[df['HomePlanet'].isnull() & (df['Cabin_cat'] == 'G'), 'HomePlanet'] = 'Earth'
test.loc[test['HomePlanet'].isnull() & test['Cabin_cat'].isin(['A', 'B', 'C', 'T']), 'HomePlanet'] = 'Europa'
test.loc[test['HomePlanet'].isnull() & (test['Cabin_cat'] == 'G'), 'HomePlanet'] = 'Earth'
df.isnull().sum()
df.loc[df['Cabin_cat'] != 'D', 'HomePlanet'] = df['HomePlanet'].fillna('Earth')
test.loc[test['Cabin_cat'] != 'D', 'HomePlanet'] = test['HomePlanet'].fillna('Earth')
df.isnull().sum()
df.loc[df['Cabin_cat'] == 'D', 'HomePlanet'] = df['HomePlanet'].fillna('Mars')
test.loc[test['Cabin_cat'] == 'D', 'HomePlanet'] = test['HomePlanet'].fillna('Mars')
df.isnull().sum()
df['Expenditure'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
test['Expenditure'] = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']
df.loc[(df['CryoSleep'] == False) & (df['Expenditure'] == 0), 'Age'] = df['Age'].fillna(5)
df['Age'] = df['Age'].fillna(df['Age'].mean())
test.loc[(test['CryoSleep'] == False) & (test['Expenditure'] == 0), 'Age'] = test['Age'].fillna(5)
test['Age'] = test['Age'].fillna(test['Age'].mean())
df['VIP'] = df['VIP'].fillna(df['VIP'].mode()[0])
test['VIP'] = test['VIP'].fillna(test['VIP'].mode()[0])
df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
test['Destination'] = test['Destination'].fillna(test['Destination'].mode()[0])
df = df.drop(['PassengerId', 'Name', 'Cabin', 'Cabin_cat', 'Age_cat', 'FoodCourt_cat', 'Spa_cat', 'VRDeck_cat', 'Expenditure', 'Group'], axis=1)
test = test.drop(['PassengerId', 'Name', 'Cabin', 'Cabin_cat', 'Age_cat', 'FoodCourt_cat', 'Spa_cat', 'VRDeck_cat', 'Expenditure', 'Group'], axis=1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['CryoSleep'] = le.fit_transform(df['CryoSleep'])
df['VIP'] = le.fit_transform(df['VIP'])
test['CryoSleep'] = le.fit_transform(test['CryoSleep'])
test['VIP'] = le.fit_transform(test['VIP'])
cols = ['HomePlanet', 'Destination']
df_oh = pd.get_dummies(df[cols], drop_first=True)
df = pd.concat([df, df_oh], axis=1)
df = df.drop(cols, axis=1)
test_oh = pd.get_dummies(test[cols], drop_first=True)
test = pd.concat([test, test_oh], axis=1)
test = test.drop(cols, axis=1)
df_y = df['Transported']
df = df.drop('Transported', axis=1)
df = pd.concat([df, df_y], axis=1)


y_df = df['Transported']
x_df = df.drop('Transported', axis=1)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x_df, y_df, test_size=0.2, random_state=156)
(x_tr, x_val, y_tr, y_val) = train_test_split(x_train, y_train, test_size=0.1, random_state=156)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
dt_clf = DecisionTreeClassifier(random_state=11, max_depth=11, min_samples_split=120)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression(solver='liblinear')