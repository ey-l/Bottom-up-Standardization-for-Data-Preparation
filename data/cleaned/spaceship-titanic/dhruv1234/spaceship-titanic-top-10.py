import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head()
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test.head()
print('Shape of Train dataset: ', train.shape)
print('Shape of Test dataset: ', test.shape)
(fig, axes) = plt.subplots(2, figsize=(10, 10))
sns.heatmap(train.isna().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data of Train dataset'}, ax=axes[0])
sns.heatmap(test.isna().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data of Test dataset'}, ax=axes[1])
train.drop(['Name'], axis=1, inplace=True)
test.drop(['Name'], axis=1, inplace=True)
cat_cols = [col for col in train.columns if train[col].dtype in (object, bool)]
for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    if col != 'Transported':
        test[col] = test[col].fillna(test[col].mode()[0])
num_cols = [col for col in train.columns if col not in cat_cols]
(fig, axes) = plt.subplots(3, 2, figsize=(10, 12))
fig.tight_layout(pad=5.0)
for (col, ax) in zip(num_cols, axes.flatten()):
    sns.kdeplot(x=col, hue='Transported', data=train, ax=ax)
    ax.set_title(col)
for col in num_cols:
    train[col] = train[col].fillna(train[col].median())
    if col != 'Transported':
        test[col] = test[col].fillna(test[col].median())
train[['Deck', 'Num', 'Side']] = train.Cabin.str.split('/', expand=True)
test[['Deck', 'Num', 'Side']] = test.Cabin.str.split('/', expand=True)
train['Num'] = train['Num'].astype('int')
test['Num'] = test['Num'].astype('int')
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
train[['group', 'number']] = train['PassengerId'].str.split('_', expand=True).astype('int')
test[['group', 'number']] = test['PassengerId'].str.split('_', expand=True).astype('int')
train.drop('PassengerId', axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)
print('Length of Groups', train['group'].shape[0])
print('Uniques of Groups', len(train['group'].unique()))
train.drop('group', axis=1, inplace=True)
test.drop('group', axis=1, inplace=True)
plt.figure(figsize=(10, 8))
values = train['Transported'].value_counts()
label = train['Transported'].value_counts().index
plt.pie(values, labels=label)
train['Transported'] = train['Transported'].astype('object')
train['CryoSleep'] = train['CryoSleep'].astype('object')
test['CryoSleep'] = test['CryoSleep'].astype('object')
train['VIP'] = train['VIP'].astype('object')
test['VIP'] = test['VIP'].astype('object')
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import OrdinalEncoder
cat_cols = [col for col in train.columns if train[col].dtype in (object, bool)]
temp = train[cat_cols]
ordi = OrdinalEncoder()
temp = ordi.fit_transform(temp)
x = temp[:, :4]
y = temp[:, -1]
chi2_model = SelectKBest(chi2, k='all')