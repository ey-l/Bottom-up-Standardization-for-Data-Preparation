import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1.head(5)
print('Titanic train dateset Shape : ', _input1.shape)
print('Titanic test dateset Shape : ', _input1.shape)
_input1.info()
_input1.describe()
Purples_palette = sns.color_palette('Purples', 10)
BuPu_palette = sns.color_palette('BuPu', 10)
sns.palplot(Purples_palette)
sns.palplot(BuPu_palette)
train_df_null_count = pd.DataFrame(_input1.isnull().sum(), columns=['Train Null count'])
test_df_null_count = pd.DataFrame(_input1.isnull().sum(), columns=['Test Null count'])
null_df = pd.concat([train_df_null_count, test_df_null_count], axis=1)
null_df.head(100).style.background_gradient(cmap='Purples')
msno.matrix(df=_input1.iloc[:, :], figsize=(5, 5), color=BuPu_palette[4])
Purples_palette_two = [Purples_palette[3], Purples_palette[6]]
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(3, 2)
ax_sex_survived = fig.add_subplot(gs[:2, :2])
sns.countplot(x='Sex', hue='Survived', data=_input1, ax=ax_sex_survived, palette=Purples_palette_two)
ax_pie_male = fig.add_subplot(gs[2, 0])
ax_pie_female = fig.add_subplot(gs[2, 1])
male = _input1[_input1['Sex'] == 'male']['Survived'].value_counts().sort_index()
ax_pie_male.pie(male, labels=male.index, autopct='%1.1f%%', explode=(0, 0.1), startangle=90, colors=Purples_palette_two)
female = _input1[_input1['Sex'] == 'female']['Survived'].value_counts().sort_index()
ax_pie_female.pie(female, labels=female.index, autopct='%1.1f%%', explode=(0, 0.1), startangle=90, colors=Purples_palette_two)
fig.text(0.25, 0.92, 'Distribution of Survived by Sex', fontweight='bold', fontfamily='serif', fontsize=17)
ax_sex_survived.patch.set_alpha(0)
pd.crosstab(_input1['Sex'], _input1['Survived'], margins=True).style.background_gradient(cmap='Purples')
pd.crosstab(_input1['Pclass'], _input1['Survived'], margins=True).style.background_gradient(cmap='BuPu')
BuPu_palette
Purples_palette_two_1 = [Purples_palette[4], Purples_palette[8]]
BuPu_palette_two = [BuPu_palette[2], BuPu_palette[4]]
(fig, ax) = plt.subplots(1, 2, figsize=(16, 8))
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=_input1, palette=BuPu_palette_two, ax=ax[0])
ax[0].patch.set_alpha(0)
ax[0].text(-0.5, 100, 'Plot showing the relationship \nbetween Pclass and Age and Survived', fontweight='bold', fontfamily='serif', fontsize=13)
sns.violinplot(x='Sex', y='Age', hue='Survived', data=_input1, palette=Purples_palette_two_1, ax=ax[1])
ax[1].set_yticks([])
ax[1].set_ylabel('')
ax[1].patch.set_alpha(0)
ax[1].text(-0.5, 100, 'Plot showing the relationship \nbetween Sex and Age and Survived', fontweight='bold', fontfamily='serif', fontsize=13)
fig.text(0.1, 1, 'Violin plot showing the relationship Age and Survived', fontweight='bold', fontfamily='serif', fontsize=20)
_input1
fig = px.scatter_3d(_input1[:1000], x='Age', y='Survived', z='Pclass', color='Age')
fig.show()
corr = _input1.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr, cmap='BuPu')
plt.title('Titanic train data Heatmap', fontweight='bold', fontsize=17)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median())
_input1['Cabin'] = _input1['Cabin'].fillna('N')
_input1['Embarked'] = _input1['Embarked'].fillna('N')
_input1.head()
_input1.loc[_input1['Fare'].isnull(), 'Fare'] = _input1['Fare'].mean()
_input1['Fare'] = _input1['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
(f, ax) = plt.subplots(1, 1, figsize=(8, 6))
sns.distplot(_input1['Fare'], label='Skewness : {:.2f}'.format(_input1['Fare'].skew()), ax=ax, color=BuPu_palette[-1])
plt.legend(loc='best')
plt.title('Check train data Skewness', fontweight='bold', fontsize=18)
ax.patch.set_alpha(0)
_input1['Cabin'] = _input1['Cabin'].str[:1]
_input1.head()
_input1 = _input1.drop(['Name', 'PassengerId', 'Ticket'], axis=1, inplace=False)
_input1.head()
_input1['Cabin'].value_counts()
Cabin_T_index = _input1[_input1['Cabin'] == 'T'].index
_input1 = _input1.drop(Cabin_T_index, inplace=False)
_input1['Embarked'].value_counts()
Embarked_N_index = _input1[_input1['Embarked'] == 'N'].index
_input1 = _input1.drop(Embarked_N_index, inplace=False)
_input1.head()
_input1 = pd.get_dummies(_input1)
_input1.head()
_input1.shape
x = _input1.drop('Survived', axis=1)
y = _input1['Survived']
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)
print('X train data size : {}'.format(x_train.shape))
print('Y train data size : {}'.format(y_train.shape))
print(' ')
print('X test data size : {}'.format(x_test.shape))
print('Y test data size : {}'.format(y_test.shape))
log_reg = LogisticRegression()