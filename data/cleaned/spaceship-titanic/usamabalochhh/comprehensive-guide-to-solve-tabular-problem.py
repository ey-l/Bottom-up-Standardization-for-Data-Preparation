import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scikitplot as skplt

class clr:
    S = '\x1b[1m' + '\x1b[94m'
    E = '\x1b[0m'
warnings.filterwarnings('ignore')
my_colors = ['#5EAFD9', '#449DD1', '#3977BB', '#2D51A5', '#5C4C8F', '#8B4679', '#C53D4C', '#E23836', '#FF4633', '#FF5746']
print(clr.S + 'Notebook Color Schemes:' + clr.E)
sns.palplot(sns.color_palette(my_colors))

train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(f'The Shpae of the Train data : {train_data.shape}')
print(f'The Shpae of the Test data : {test_data.shape}')
train_data.head()

def get_csv(df, name):
    print(clr.S + f' === {name} === ' + clr.E)
    print(clr.S + 'Total Missing Values : ' + clr.E, df.isnull().sum().sum())
    print(clr.S + 'Columns = ' + clr.E, list(df.columns), '\n\n')
get_csv(train_data, 'Train')
get_csv(test_data, 'Test')
train_data.info()
train_data.describe().T
plt.figure(figsize=(22, 24))
plt.subplot(4, 3, 1)
train_data.dtypes.value_counts().plot(kind='pie', autopct='%.1f%%', colors=[my_colors[4], my_colors[5], my_colors[6]])
plt.subplot(4, 3, 2)
sns.countplot(data=train_data, x='Transported', palette=[my_colors[1], my_colors[2]])
print(clr.S + " --- Count of Nan's in Every Column --- " + clr.E)
train_data.isnull().sum()
plt.figure(figsize=(6, 4))
plt.title('Count of Nan in Every Columns')
train_data.isnull().sum().plot(kind='bar', color=my_colors[3])
print(clr.S + 'Percentage of Nan columns in Train data ' + clr.E)
round(train_data.isnull().sum() / len(train_data) * 100, 2)
num_data = train_data.select_dtypes(exclude=['object']).copy()
num_data.head()
plt.figure(figsize=(22, 24))
temp = num_data['Age']
x = pd.Series(temp, name='Age Variable')
plt.subplot(4, 3, 1)
ax = sns.distplot(temp, bins=10, color=my_colors[0])
ax.set_title('Distribution of Age Variable')
plt.subplot(4, 3, 2)
ax = sns.kdeplot(x, shade=True, color=my_colors[8])
ax.set_title('Distribution of Age Variable')

plt.figure(figsize=(10, 4))
ax = sns.boxplot(num_data['Age'], color=my_colors[4])
ax.set_title('Visualizing Age Column to detect Outliers')

plt.figure(figsize=(22, 24))
temp = num_data['RoomService']
x = pd.Series(temp, name='RoomService Variable')
plt.subplot(4, 3, 1)
ax = sns.distplot(temp, bins=10, color=my_colors[0])
ax.set_title('Distribution of RoomService Variable')
plt.subplot(4, 3, 2)
ax = sns.kdeplot(x, shade=True, color=my_colors[2])
ax.set_title('Distribution of RoomService Variable')

plt.figure(figsize=(10, 4))
ax = sns.boxplot(num_data['RoomService'], color=my_colors[5])
ax.set_title('Visualizing RoomService Column to detect Outliers')

plt.figure(figsize=(22, 24))
temp = num_data['FoodCourt']
x = pd.Series(temp, name='FoodCourt Variable')
plt.subplot(4, 3, 1)
ax = sns.distplot(temp, bins=10, color=my_colors[0])
ax.set_title('Distribution of FoodCourt Variable')
plt.subplot(4, 3, 2)
ax = sns.kdeplot(x, shade=True, color=my_colors[9])
ax.set_title('Distribution of FoodCourt Variable')

plt.figure(figsize=(10, 4))
ax = sns.boxplot(num_data['FoodCourt'], color=my_colors[7])
ax.set_title('Visualizing FoodCourt Column to detect Outliers')

plt.figure(figsize=(22, 24))
temp = num_data['Spa']
x = pd.Series(temp, name='Spa Variable')
plt.subplot(4, 3, 1)
ax = sns.distplot(temp, bins=10, color=my_colors[0])
ax.set_title('Distribution of Spa Variable')
plt.subplot(4, 3, 2)
ax = sns.kdeplot(x, shade=True, color=my_colors[8])
ax.set_title('Distribution of Spa Variable')

plt.figure(figsize=(10, 4))
ax = sns.boxplot(num_data['Spa'], color=my_colors[8])
ax.set_title('Visualizing Spa Column to detect Outliers')

plt.figure(figsize=(22, 24))
temp = num_data['ShoppingMall']
x = pd.Series(temp, name='ShoppingMall Variable')
plt.subplot(4, 3, 1)
ax = sns.distplot(temp, bins=10, color=my_colors[0])
ax.set_title('Distribution of ShoppingMall Variable')
plt.subplot(4, 3, 2)
ax = sns.kdeplot(x, shade=True, color=my_colors[8])
ax.set_title('Distribution of ShoppingMall Variable')

plt.figure(figsize=(10, 4))
ax = sns.boxplot(num_data['ShoppingMall'], color=my_colors[9])
ax.set_title('Visualizing ShoppingMall Column to detect Outliers')

plt.figure(figsize=(22, 24))
temp = num_data['VRDeck']
x = pd.Series(temp, name='VRDeck Variable')
plt.subplot(4, 3, 1)
ax = sns.distplot(temp, bins=10, color=my_colors[0])
ax.set_title('Distribution of VRDeck Variable')
plt.subplot(4, 3, 2)
ax = sns.kdeplot(x, shade=True, color=my_colors[4])
ax.set_title('Distribution of VRDeck Variable')

plt.figure(figsize=(10, 4))
ax = sns.boxplot(num_data['VRDeck'], color=my_colors[9])
ax.set_title('Visualizing VRDeck Column to detect Outliers')

plt.figure(figsize=(12, 6))
temp = num_data['Age']
x = pd.Series(temp, name='Age Variable')
ax = sns.histplot(temp, bins=10, color=my_colors[3])
ax.set_title('Distribution of the Age', fontsize=18)
style = 'Simple, tail_width=1, head_width=12, head_length=14'
kw = dict(arrowstyle=style, color=my_colors[9])
arrow = patches.FancyArrowPatch((45, 1600), (24, 1100), connectionstyle='arc3,rad=-.10', **kw)
plt.gca().add_patch(arrow)
plt.text(x=40, y=1700, s=f'Most of the People are from 18-30', color='black', size=14)
num_data['child'] = num_data['Age'].apply(lambda x: 1 if x >= 1 and x <= 14 else 0)
num_data['youth'] = num_data['Age'].apply(lambda x: 1 if x >= 15 and x <= 24 else 0)
num_data['adult'] = num_data['Age'].apply(lambda x: 1 if x >= 25 and x <= 64 else 0)
num_data['senior'] = num_data['Age'].apply(lambda x: 1 if x >= 65 else 0)
num_data['Transported'] = num_data['Transported'].astype(int)
print(clr.S + ' === Checking Age Group === ' + clr.E, '\n')
print(f"There are : {sum(num_data['child'])} Childrens in Spaceship")
print(f"There are : {sum(num_data['youth'])} Youths in Spaceship")
print(f"There are : {sum(num_data['adult'])} Adult in Spaceship")
print(f"There are : {sum(num_data['senior'])} Senior in Spaceship\n\n")
(fig, ax) = plt.subplots(4, 1, figsize=(6, 16))
sns.countplot(x='Transported', hue='child', data=num_data, ax=ax[0], palette=[my_colors[6], my_colors[9]])
ax[0].set_title('Child Group with Respect to Transported')
sns.countplot(x='Transported', hue='youth', data=num_data, ax=ax[1], palette=[my_colors[4], my_colors[3]])
ax[1].set_title('Youth Group with Respect to Transported')
sns.countplot(x='Transported', hue='adult', data=num_data, ax=ax[2], palette=[my_colors[2], my_colors[0]])
ax[2].set_title('Adult Group with Respect to Transported')
sns.countplot(x='Transported', hue='senior', data=num_data, ax=ax[3], palette=[my_colors[0], my_colors[3]])
ax[3].set_title('Senior Group with Respect to Transported')
plt.tight_layout(pad=3.0)

num_data['Total_Expense'] = num_data['RoomService'] + num_data['Spa'] + num_data['ShoppingMall'] + num_data['VRDeck'] + num_data['FoodCourt']
plt.figure(figsize=(10, 4))
x = num_data['Total_Expense']
ax = sns.distplot(x, kde=True, bins=10, color=my_colors[0])
ax.set_title('Total Expense Distribution')
style = 'Simple, tail_width=1, head_width=12, head_length=14'
kw = dict(arrowstyle=style, color=my_colors[5])
arrow = patches.FancyArrowPatch((30000, 0.0002), (15000, 0.0001), connectionstyle='arc3,rad=-.10', **kw)
plt.gca().add_patch(arrow)
plt.text(x=15200, y=0.00025, s=f'Expense More then this Threshold can be Summed with Outliers', color='black', size=10)
plt.axvline(x=15000, linestyle='--', color='black')

cat_data = train_data.select_dtypes(exclude=['float']).copy()
cat_data.head()
cat_data.isnull().sum()
cat_data.nunique()
for col in cat_data.columns:
    print(cat_data[col].value_counts())
cat_data['Transported'].unique()
print(clr.S + 'Percentage of True and False' + clr.E)
print(round(cat_data['Transported'].value_counts() / len(cat_data['Transported']), 3))
(fig, ax) = plt.subplots(1, 2, figsize=(12, 6))
ax[0] = cat_data['Transported'].value_counts().plot(kind='pie', autopct='%.1f%%', ax=ax[0], colors=[my_colors[0], my_colors[4]], shadow=True)
ax[0].set_title('Transported Percentage')
ax[1] = sns.countplot(x='Transported', data=cat_data, palette=[my_colors[9], my_colors[6]])
ax[1].set_title('Transported Counts')

(f, ax) = plt.subplots(figsize=(10, 6))
ax = sns.countplot(x='Transported', hue='HomePlanet', data=cat_data, palette='Set1')
ax.set_title('Frequencey distribution of HomePlanet with respect to Transported')

(f, ax) = plt.subplots(figsize=(10, 6))
ax = sns.countplot(x='Transported', hue='CryoSleep', data=cat_data, palette='Set2')
ax.set_title('Frequency distribution of CryoSleep Column with respect to Transported', fontsize=16)

(f, ax) = plt.subplots(figsize=(10, 6))
ax = sns.countplot(x='Transported', hue='Destination', data=cat_data, palette='Set2')
ax.set_title('Frequency distribution of Destination Column with respect to Transported', fontsize=16)

(f, ax) = plt.subplots(figsize=(10, 6))
ax = sns.countplot(x='Transported', hue='VIP', data=cat_data, palette='Set1')
ax.set_title('Frequency distribution of VIP Column with respect to Transported', fontsize=16)

cat_data['Cabin'] = cat_data['Cabin'].fillna(cat_data['Cabin'].mode()[0])
cat_data['Deck'] = cat_data['Cabin'].apply(lambda x: x.split('/')[0])
cat_data['num'] = cat_data['Cabin'].apply(lambda x: x.split('/')[1])
cat_data['side'] = cat_data['Cabin'].apply(lambda x: x.split('/')[2])
print(cat_data['Deck'].value_counts())
print(cat_data['num'].value_counts())
print(cat_data['side'].value_counts())
(fig, ax) = plt.subplots(figsize=(10, 6))
ax = sns.countplot(x='Deck', data=cat_data, palette='Set1')
ax.set_title('Frequency distribution of Deck')

(fig, ax) = plt.subplots(figsize=(10, 6))
ax = sns.countplot(x='Transported', hue='Deck', data=cat_data, palette='Set1')
ax.set_title('Frequency distribution of Deck with respect to Transported')

(fig, ax) = plt.subplots(figsize=(10, 6))
ax = sns.countplot(x='HomePlanet', hue='Deck', data=cat_data, palette='Set1')
ax.set_title('Frequency distribution of Deck with respect to HomePlanet')

(fig, ax) = plt.subplots(figsize=(8, 4))
ax = sns.countplot(x='side', data=cat_data, palette='Set3')
ax.set_title('Frequency distribution of side')

(fig, ax) = plt.subplots(figsize=(10, 6))
ax = sns.countplot(x='Transported', hue='side', data=cat_data, palette='Set2')
ax.set_title('Frequency distribution of side with respect to Transported')

(fig, ax) = plt.subplots(figsize=(10, 6))
ax = sns.countplot(x='Deck', hue='side', data=cat_data, palette='Set1')
ax.set_title('Frequency distribution of Side with respect of Deck')

cat_data['group_id'] = cat_data['PassengerId'].apply(lambda x: x.split('_')[0])
cat_data['No_of_family_members'] = cat_data['group_id'].map(cat_data['group_id'].value_counts())
print(cat_data['No_of_family_members'])
le = LabelEncoder()
print(pd.get_dummies(cat_data['Deck']))
print(le.fit_transform(cat_data['side']))
print(le.fit_transform(cat_data['HomePlanet']))
print(le.fit_transform(cat_data['Destination']))
train_data.isnull().sum()

def preprocess(df, name):
    df.drop('Name', axis=1, inplace=True)
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['RoomService'] = df['RoomService'].fillna(df['RoomService'].median())
    df['Spa'] = df['Spa'].fillna(df['Spa'].median())
    df['FoodCourt'] = df['FoodCourt'].fillna(df['FoodCourt'].median())
    df['ShoppingMall'] = df['ShoppingMall'].fillna(df['ShoppingMall'].median())
    df['VRDeck'] = df['VRDeck'].fillna(df['VRDeck'].median())
    df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode()[0])
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
    df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])
    df['VIP'] = df['VIP'].fillna(df['VIP'].mode()[0])
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
    if name == 'Train':
        df.drop(df[df['RoomService'] > 10000].index, axis=0, inplace=True)
        df.drop(df[df['VRDeck'] > 20000].index, axis=0, inplace=True)
        df.drop(df[df['Spa'] > 20000].index, axis=0, inplace=True)
        df.drop(df[df['FoodCourt'] > 20000].index, axis=0, inplace=True)
        df.drop(df[df['ShoppingMall'] > 20000].index, axis=0, inplace=True)
    return df
train_data = preprocess(train_data, 'Train')
test_data = preprocess(test_data, 'Test')
print(clr.S + 'Missing Train Data ' + clr.E, train_data.isnull().sum().sum())
print(clr.S + 'Missing Test Data ' + clr.E, test_data.isnull().sum().sum())

def feature_engineering(df):
    df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0])
    df['num'] = df['Cabin'].apply(lambda x: x.split('/')[1])
    df['side'] = df['Cabin'].apply(lambda x: x.split('/')[2])
    df['child'] = df['Age'].apply(lambda x: 1 if x >= 1 and x <= 14 else 0)
    df['youth'] = df['Age'].apply(lambda x: 1 if x >= 15 and x <= 24 else 0)
    df['adult'] = df['Age'].apply(lambda x: 1 if x >= 25 and x <= 64 else 0)
    df['senior'] = df['Age'].apply(lambda x: 1 if x >= 65 else 0)
    df['Total_Expense'] = df['RoomService'] + df['Spa'] + df['ShoppingMall'] + df['VRDeck'] + df['FoodCourt']
    df['group_id'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['No_of_family_members'] = df['group_id'].map(df['group_id'].value_counts())
    le = LabelEncoder()
    df['CryoSleep'] = le.fit_transform(df['CryoSleep'])
    df['VIP'] = le.fit_transform(df['VIP'])
    df['HomePlanet'] = le.fit_transform(df['HomePlanet'])
    df['Destination'] = le.fit_transform(df['Destination'])
    df['side'] = le.fit_transform(df['side'])
    df['num'] = le.fit_transform(df['num'])
    dff = pd.get_dummies(df['Deck'])
    dff['PassengerId'] = df['PassengerId']
    df = pd.merge(df, dff, on='PassengerId')
    return df
train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

def remove_feature(df, thresh):
    corr_col = set()
    correlation = df.corr()
    for i in range(len(correlation.columns)):
        for j in range(i):
            if abs(correlation.iloc[i, j]) > thresh:
                column = correlation.columns[i]
                corr_col.add(column)
    return corr_col
correlated_features = remove_feature(train_data, 0.8)
print(f'Highly correlated Features : {correlated_features}')
train_data.drop(['PassengerId', 'Cabin', 'group_id', 'Deck', 'Age'], axis=1, inplace=True)
test_data.drop(['PassengerId', 'Cabin', 'group_id', 'Deck', 'Age'], axis=1, inplace=True)
train_data.head()

def Scaling(df, name):
    if name == 'train':
        t_data = df.copy().drop('Transported', axis=1)
        ss = StandardScaler()