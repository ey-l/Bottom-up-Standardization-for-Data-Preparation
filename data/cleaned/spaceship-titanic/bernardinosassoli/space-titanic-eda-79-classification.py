import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

(train_data, test_data) = (pd.read_csv('data/input/spaceship-titanic/train.csv'), pd.read_csv('data/input/spaceship-titanic/test.csv'))
train_data.head()
train_data.info()
train_data.describe()
train_data.isnull().sum()
train_data[train_data['Cabin'].isnull() == True].head()

def fill_proportionally(col, dataset):
    values = dataset[col].dropna().unique()
    weights = dataset[col].value_counts().values / dataset[col].value_counts().values.sum()
    dataset[col] = dataset[col].apply(lambda x: random.choices(values, weights=weights)[0] if pd.isnull(x) else x)
    assert dataset[col].isna().sum() == 0
for column in ['Destination', 'HomePlanet', 'CryoSleep']:
    fill_proportionally(column, train_data)
    fill_proportionally(column, test_data)
train_data.isna().sum()
for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']:
    train_data[col].fillna(train_data[col].median(), inplace=True)
    test_data[col].fillna(train_data[col].median(), inplace=True)
test_data['VIP'].fillna(False, inplace=True)
train_data['VIP'].fillna(False, inplace=True)
print(train_data.isna().sum())
print('Missing data %: ', train_data.isna().sum().sum() / train_data.shape[0])
train_data['Name'].fillna('', inplace=True)
(train_data.shape, train_data.isna().sum().sum())
train_data['Cabin'].fillna('Z/0000/Z', inplace=True)
test_data['Cabin'].fillna('Z/0000/Z', inplace=True)
train_data['Cabin'].value_counts(dropna=False)
train_data[train_data['Cabin'] == 'G/734/S']
train_data['Deck'] = train_data['Cabin'].apply(lambda x: str(x)[0])
test_data['Deck'] = test_data['Cabin'].apply(lambda x: str(x)[0])
train_data['Deck'].value_counts(dropna=False)
test_data['Deck'].value_counts(dropna=False)
train_data['Side'] = train_data['Cabin'].apply(lambda x: str(x).split('/')[2])
test_data['Side'] = test_data['Cabin'].apply(lambda x: str(x).split('/')[2])
print(test_data['Side'].value_counts(dropna=False))
print(train_data['Side'].value_counts(dropna=False))
train_data['GroupId'] = train_data['PassengerId'].apply(lambda x: x.split('_')[0])
test_data['GroupId'] = test_data['PassengerId'].apply(lambda x: x.split('_')[0])
train_data['GroupIdProgNumber'] = train_data['PassengerId'].apply(lambda x: x.split('_')[1])
test_data['GroupIdProgNumber'] = test_data['PassengerId'].apply(lambda x: x.split('_')[1])
groups = train_data[train_data['GroupId'].duplicated()]['GroupId']
train_data['InGroup'] = train_data['GroupId'].apply(lambda x: x in groups.values)
groups = test_data[test_data['GroupId'].duplicated()]['GroupId']
test_data['InGroup'] = test_data['GroupId'].apply(lambda x: x in groups.values)
train_data['InGroup'].value_counts()
train_data['GroupSize'] = train_data['GroupId'].apply(lambda x: train_data['GroupId'].value_counts().loc[x])
test_data['GroupSize'] = test_data['GroupId'].apply(lambda x: test_data['GroupId'].value_counts().loc[x])
train_data['GroupSize'].value_counts()
columns_to_plot = ['Destination', 'VIP', 'HomePlanet', 'InGroup', 'CryoSleep', 'Transported']
rows = 3
columns = 2
ix = 0
(fig, axes) = plt.subplots(rows, columns, figsize=(9, 7))
for row in range(rows):
    for col in range(columns):
        try:
            sns.countplot(data=train_data, x=columns_to_plot[ix], ax=axes[row][col])
            sns.despine()
            ix += 1
        except Exception:
            axes[row][col].set_visible(False)
plt.tight_layout()
sns.countplot(data=train_data, x='HomePlanet', hue='Transported')
sns.countplot(data=train_data, x='Destination', hue='Transported')
sns.countplot(data=train_data, x='VIP', hue='Transported')
sns.countplot(data=train_data, x='InGroup', hue='Transported')
sns.countplot(data=train_data, x='CryoSleep', hue='Transported')
sns.countplot(x='Side', data=train_data, hue='Transported')
sns.countplot(x='Deck', data=train_data, hue='Transported')
train_data['Deck'] = train_data['Deck'].apply(lambda x: 'F' if x == 'T' else x)
test_data['Deck'] = test_data['Deck'].apply(lambda x: 'F' if x == 'T' else x)
sns.catplot(x='Deck', data=train_data, hue='Transported', col='Side', kind='count')
sns.despine()
sns.countplot(x='GroupSize', data=train_data, hue='Transported')
sns.kdeplot(data=train_data, x='Age', hue='Transported', fill=True)
plt.title('Age distribution')
(min_age, max_age) = (train_data['Age'].min(), train_data['Age'].max())
bins = np.linspace(min_age, max_age, 6)
print(bins)
labels = ['Child', 'Young', 'Middle', 'Senior', 'Elder']
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=bins, labels=labels, include_lowest=True)
sns.countplot(data=train_data, x='AgeGroup', hue='Transported')
test_data['AgeGroup'] = pd.cut(test_data['Age'], bins=bins, labels=labels, include_lowest=True)
train_data['all'] = ''
sns.violinplot(data=train_data, y='Age', x='all', hue='Transported', split=True)
train_data.drop('all', axis=1, inplace=True)
data_to_plot = train_data.describe().columns
rows = 3
cols = 2
(fig, axes) = plt.subplots(rows, cols, figsize=(12, 8))
ix = 0
for i in range(rows):
    for j in range(cols):
        sns.kdeplot(x=data_to_plot[ix], ax=axes[i][j], hue='Transported', data=train_data, fill=True)
        sns.despine()
        ix += 1
plt.tight_layout()
data_to_plot = train_data.describe().columns
rows = 3
cols = 2
(fig, axes) = plt.subplots(rows, cols, figsize=(12, 8))
ix = 0
for i in range(rows):
    for j in range(cols):
        sns.boxenplot(x=data_to_plot[ix], ax=axes[i][j], data=train_data)
        sns.despine()
        ix += 1
plt.tight_layout()
sns.kdeplot(data=train_data, x='Spa')
capped = train_data.copy()
upper_limit = train_data['RoomService'].quantile(0.75)
lower_limit = test_data['RoomService'].quantile(0.25)
iqr = upper_limit - lower_limit
upper_limit += iqr * 1.5
lower_limit -= iqr * 1.5
capped['RoomService'] = np.where(capped['RoomService'] > upper_limit, upper_limit, capped['RoomService'])
capped['RoomService'] = np.where(capped['Spa'] < lower_limit, lower_limit, capped['RoomService'])
print(capped['RoomService'].skew(), train_data['RoomService'].skew())
(fig, axes) = plt.subplots(1, 2)
sns.kdeplot(data=capped, x='RoomService', ax=axes[0])
axes[0].set_title('With outliers')
sns.kdeplot(data=train_data, x='RoomService', ax=axes[1])
axes[1].set_title('Without outliers')
plt.tight_layout()
sns.boxenplot(data=capped, x='RoomService')
"\nfor col in ['RoomService','FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:\n    upper_limit = train_data[col].quantile(0.75)\n    lower_limit = test_data[col].quantile(0.25)\n    iqr = upper_limit - lower_limit\n    upper_limit += iqr * 1.5\n    lower_limit -= iqr * 1.5\n    train_data[col] = np.where(train_data[col] > upper_limit, upper_limit, train_data[col])\n    train_data[col] = np.where(train_data[col] < lower_limit, lower_limit, train_data[col])\n"
"\nfor col in ['RoomService','FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:\n    train_data[col] = np.log(1 + train_data[col])\n    test_data[col] = np.log(1+ test_data[col])\n"
train_data = train_data[['HomePlanet', 'CryoSleep', 'Destination', 'AgeGroup', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'Side', 'InGroup', 'GroupSize', 'Age', 'Transported']]
test_data = test_data[['HomePlanet', 'CryoSleep', 'Destination', 'AgeGroup', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'Side', 'InGroup', 'GroupSize', 'Age']]
models = []
train_dataset = train_data.copy()
test_dataset = test_data.copy()
categoricals = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'InGroup', 'AgeGroup']
numericals = ['VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupSize', 'Age']
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_curve
(train_data.columns, test_data.columns)
train_data.head()
transformer = ColumnTransformer([('num', StandardScaler(), numericals), ('cat', OneHotEncoder(), categoricals)])
pipeline = Pipeline([('transformer', transformer), ('classifier', LogisticRegression(max_iter=500))])
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
X
cv = KFold(n_splits=10, random_state=42, shuffle=True)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean accuracy of K-fold cross validation: {:.2f} %'.format(np.mean(scores) * 100))
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)