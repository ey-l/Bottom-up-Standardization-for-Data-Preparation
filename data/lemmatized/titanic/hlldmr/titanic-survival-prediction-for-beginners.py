import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split, GridSearchCV
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
train = _input1.copy()
test = _input0.copy()
train.describe(include='all')
train.info()
print(train.columns)
train.head()
test.head()
train.tail()
train.sample(5)
print(pd.isnull(train).sum())
train.describe().T
100 * train.isnull().sum() / len(train)
train['Pclass'].value_counts()
train['Sex'].value_counts()
train['SibSp'].value_counts()
train['Parch'].value_counts()
train['Ticket'].value_counts()
train['Cabin'].value_counts()
train['Embarked'].value_counts()
sns.barplot(x='Sex', y='Survived', data=train)
print('Percentage of females who survived:', train['Survived'][train['Sex'] == 'female'].value_counts(normalize=True)[1] * 100)
print('Percentage of males who survived:', train['Survived'][train['Sex'] == 'male'].value_counts(normalize=True)[1] * 100)
sns.barplot(x='Pclass', y='Survived', data=train)
print('Percentage of Pclass = 1 who survived:', train['Survived'][train['Pclass'] == 1].value_counts(normalize=True)[1] * 100)
print('Percentage of Pclass = 2 who survived:', train['Survived'][train['Pclass'] == 2].value_counts(normalize=True)[1] * 100)
print('Percentage of Pclass = 3 who survived:', train['Survived'][train['Pclass'] == 3].value_counts(normalize=True)[1] * 100)
sns.barplot(x='SibSp', y='Survived', data=train)
print('Percentage of SibSp = 0 who survived:', train['Survived'][train['SibSp'] == 0].value_counts(normalize=True)[1] * 100)
print('Percentage of SibSp = 1 who survived:', train['Survived'][train['SibSp'] == 1].value_counts(normalize=True)[1] * 100)
print('Percentage of SibSp = 2 who survived:', train['Survived'][train['SibSp'] == 2].value_counts(normalize=True)[1] * 100)
print('Percentage of SibSp = 3 who survived:', train['Survived'][train['SibSp'] == 3].value_counts(normalize=True)[1] * 100)
print('Percentage of SibSp = 4 who survived:', train['Survived'][train['SibSp'] == 4].value_counts(normalize=True)[1] * 100)
sns.barplot(x='Parch', y='Survived', data=train)
train['CabinBool'] = train['Cabin'].notnull().astype('int')
test['CabinBool'] = test['Cabin'].notnull().astype('int')
print('Percentage of CabinBool = 1 who survived:', train['Survived'][train['CabinBool'] == 1].value_counts(normalize=True)[1] * 100)
print('Percentage of CabinBool = 0 who survived:', train['Survived'][train['CabinBool'] == 0].value_counts(normalize=True)[1] * 100)
sns.barplot(x='CabinBool', y='Survived', data=train)
test.describe().T
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
train.head()
train.isnull().sum()
test.isnull().sum()
print(pd.isnull(train.CabinBool).sum())
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)
train.head()
train.describe().T
sns.boxplot(x=train['Fare'])
Q1 = train['Fare'].quantile(0.05)
Q3 = train['Fare'].quantile(0.95)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
lower_limit
upper_limit = Q3 + 1.5 * IQR
upper_limit
train['Fare'] > upper_limit
train.sort_values('Fare', ascending=False).head()
train['Fare'] = train['Fare'].replace(512.3292, 270)
train.sort_values('Fare', ascending=False).head()
train.sort_values('Fare', ascending=False)
test.sort_values('Fare', ascending=False)
test['Fare'] = test['Fare'].replace(512.3292, 270)
test.sort_values('Fare', ascending=False)
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)
train.describe().T
train.isnull().sum()
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
train.isnull().sum()
test.isnull().sum()
train.describe().T
train.isnull().sum()
test.isnull().sum()
print('Number of people embarking in Southampton (S):')
southampton = train[train['Embarked'] == 'S'].shape[0]
print(southampton)
print('Number of people embarking in Cherbourg (C):')
cherbourg = train[train['Embarked'] == 'C'].shape[0]
print(cherbourg)
print('Number of people embarking in Queenstown (Q):')
queenstown = train[train['Embarked'] == 'Q'].shape[0]
print(queenstown)
train['Embarked'].value_counts()
train = train.fillna({'Embarked': 'S'})
test = test.fillna({'Embarked': 'S'})
train.Embarked
train.isnull().sum()
test.isnull().sum()
print(pd.isnull(train.Embarked).sum())
test[test['Fare'].isnull()]
test[['Pclass', 'Fare']].groupby('Pclass').mean()
test['Fare'] = test['Fare'].fillna(12)
test['Fare'].isnull().sum()
train.head()
test.head()
sex_mapping = {'male': 0, 'female': 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)
train.head()
from sklearn import preprocessing
lbe = preprocessing.LabelEncoder()
train['Embarked'] = lbe.fit_transform(train['Embarked'])
test['Embarked'] = lbe.fit_transform(test['Embarked'])
train.head()
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels=mylabels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels=mylabels)
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train.head()
train = train.drop(['Age'], axis=1)
test = test.drop(['Age'], axis=1)
train.head()
train['FareBand'] = pd.qcut(train['Fare'], 4, labels=[1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4])
train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)
train.head()
train.head()
train['FamilySize'] = _input1['SibSp'] + _input1['Parch'] + 1
test['FamilySize'] = _input0['SibSp'] + _input0['Parch'] + 1
train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if s == 2 else 0)
train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
train.head()
test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)
test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if s == 2 else 0)
test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
test.head()
train = pd.get_dummies(train, columns=['Embarked'], prefix='Em')
train.head()
test = pd.get_dummies(test, columns=['Embarked'], prefix='Em')
test.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train['Survived']
(x_train, x_test, y_train, y_test) = train_test_split(predictors, target, test_size=0.2, random_state=0)
x_train.shape
x_test.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()