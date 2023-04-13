import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
print('Shape of Train dataset: ', _input1.shape)
print('Shape of Test dataset: ', _input0.shape)
(fig, axes) = plt.subplots(2, figsize=(10, 10))
sns.heatmap(_input1.isna().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data of Train dataset'}, ax=axes[0])
sns.heatmap(_input0.isna().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data of Test dataset'}, ax=axes[1])
_input1 = _input1.drop(['Name'], axis=1, inplace=False)
_input0 = _input0.drop(['Name'], axis=1, inplace=False)
cat_cols = [col for col in _input1.columns if _input1[col].dtype in (object, bool)]
for col in cat_cols:
    _input1[col] = _input1[col].fillna(_input1[col].mode()[0])
    if col != 'Transported':
        _input0[col] = _input0[col].fillna(_input0[col].mode()[0])
num_cols = [col for col in _input1.columns if col not in cat_cols]
(fig, axes) = plt.subplots(3, 2, figsize=(10, 12))
fig.tight_layout(pad=5.0)
for (col, ax) in zip(num_cols, axes.flatten()):
    sns.kdeplot(x=col, hue='Transported', data=_input1, ax=ax)
    ax.set_title(col)
for col in num_cols:
    _input1[col] = _input1[col].fillna(_input1[col].median())
    if col != 'Transported':
        _input0[col] = _input0[col].fillna(_input0[col].median())
_input1[['Deck', 'Num', 'Side']] = _input1.Cabin.str.split('/', expand=True)
_input0[['Deck', 'Num', 'Side']] = _input0.Cabin.str.split('/', expand=True)
_input1['Num'] = _input1['Num'].astype('int')
_input0['Num'] = _input0['Num'].astype('int')
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
_input1[['group', 'number']] = _input1['PassengerId'].str.split('_', expand=True).astype('int')
_input0[['group', 'number']] = _input0['PassengerId'].str.split('_', expand=True).astype('int')
_input1 = _input1.drop('PassengerId', axis=1, inplace=False)
_input0 = _input0.drop('PassengerId', axis=1, inplace=False)
print('Length of Groups', _input1['group'].shape[0])
print('Uniques of Groups', len(_input1['group'].unique()))
_input1 = _input1.drop('group', axis=1, inplace=False)
_input0 = _input0.drop('group', axis=1, inplace=False)
plt.figure(figsize=(10, 8))
values = _input1['Transported'].value_counts()
label = _input1['Transported'].value_counts().index
plt.pie(values, labels=label)
_input1['Transported'] = _input1['Transported'].astype('object')
_input1['CryoSleep'] = _input1['CryoSleep'].astype('object')
_input0['CryoSleep'] = _input0['CryoSleep'].astype('object')
_input1['VIP'] = _input1['VIP'].astype('object')
_input0['VIP'] = _input0['VIP'].astype('object')
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import OrdinalEncoder
cat_cols = [col for col in _input1.columns if _input1[col].dtype in (object, bool)]
temp = _input1[cat_cols]
ordi = OrdinalEncoder()
temp = ordi.fit_transform(temp)
x = temp[:, :4]
y = temp[:, -1]
chi2_model = SelectKBest(chi2, k='all')