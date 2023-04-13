import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
plt.style.use('fivethirtyeight')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.info()
_input0.info()
_input1.head()
_input1.columns = [x.lower() for x in _input1.columns]
_input1.columns
_input0.columns = [x.lower() for x in _input0.columns]
_input1 = _input1.rename(columns={'passengerid': 'passenger_id', 'pclass': 'passenger_class', 'sibsp': 'sibling_spouse', 'parch': 'parent_children'}, inplace=False)
_input0 = _input0.rename(columns={'passengerid': 'passenger_id', 'pclass': 'passenger_class', 'sibsp': 'sibling_spouse', 'parch': 'parent_children'}, inplace=False)
_input1.head()
_input1.isnull().sum()
_input1.isnull().sum().plot(kind='bar')
sns.heatmap(_input1.isnull(), cbar=False)
_input1[['passenger_id']]
plt.figure(figsize=(12, 5))
g = sns.FacetGrid(_input1, col='survived', size=5)
g = g.map(sns.distplot, 'passenger_id')
_input1.passenger_class.unique()
_input1.passenger_class.value_counts().plot(kind='pie')
_input1.passenger_class.value_counts().plot(kind='bar')
plt.figure(figsize=(12, 5))
sns.countplot('passenger_class', data=_input1, hue='survived', palette='hls')
plt.ylabel('Count', fontsize=18)
plt.xlabel('P Class', fontsize=18)
plt.title('P Class Distribution ', fontsize=20)
_input1.groupby('passenger_class').survived.value_counts(normalize=True).sort_index()
_input1.name.unique()
_input1.name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
_input1['salutation'] = _input1.name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
_input0['salutation'] = _input0.name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
_input1.salutation.value_counts()
plt.figure(figsize=(16, 5))
sns.countplot(x='salutation', data=_input1)
plt.xlabel('Salutation', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Salutation Count', fontsize=20)
plt.xticks(rotation=45)
salutation_dict = {'Capt': '0', 'Col': '0', 'Major': '0', 'Dr': '0', 'Rev': '0', 'Jonkheer': '1', 'Don': '1', 'Sir': '1', 'the Countess': '1', 'Dona': '1', 'Lady': '1', 'Mme': '2', 'Ms': '2', 'Mrs': '2', 'Mlle': '3', 'Miss': '3', 'Mr': '4', 'Master': '5'}
_input1['salutation'] = _input1.salutation.map(salutation_dict)
_input0['salutation'] = _input0.salutation.map(salutation_dict)
plt.figure(figsize=(16, 5))
sns.countplot(x='salutation', data=_input1)
plt.xlabel('Salutation', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Salutation Count', fontsize=20)
plt.xticks(rotation=45)
_input1.salutation = _input1.salutation.astype('float64')
_input0.salutation = _input0.salutation.astype('float64')
_input1.salutation.value_counts().plot(kind='pie')
plt.figure(figsize=(16, 5))
sns.countplot(x='salutation', data=_input1, hue='survived')
plt.xlabel('Salutation', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Salutation Count', fontsize=20)
plt.xticks(rotation=45)
_input1.groupby('salutation').survived.value_counts(normalize=True).sort_index()
_input1.groupby('salutation').survived.value_counts(normalize=True).sort_index().unstack()
sal_sur_index = _input1[_input1.salutation.isin([1.0, 2.0, 3.0, 5.0])].index
sal_sur_index_test = _input0[_input0.salutation.isin([1.0, 2.0, 3.0, 5.0])].index
_input1['sal_sur'] = 0
_input1.loc[sal_sur_index, 'sal_sur'] = 1
_input0['sal_sur'] = 0
_input0.loc[sal_sur_index_test, 'sal_sur'] = 1
_input1[['sal_sur', 'survived']].head()
plt.figure(figsize=(16, 5))
sns.countplot(x='sal_sur', data=_input1, hue='survived')
plt.xlabel('Salutation Survived', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Salutation Survived Count', fontsize=20)
plt.xticks(rotation=45)
_input1.sex.unique()
_input1.sex.value_counts(normalize=True)
_input1.sex.value_counts().plot(kind='pie')
_input1.sex.value_counts().plot(kind='bar')
plt.figure(figsize=(12, 5))
sns.countplot('sex', data=_input1, hue='survived', palette='hls')
plt.ylabel('Count', fontsize=18)
plt.xlabel('Sex', fontsize=18)
plt.title('Sex Distribution ', fontsize=20)
_input1.groupby('sex').survived.value_counts(normalize=True).sort_index()
_input1[['sex', 'survived']].groupby(['sex'], as_index=False).mean().sort_values(by='survived', ascending=False)
_input1.age.isnull().sum()
age_group = _input1.groupby(['sex', 'passenger_class', 'salutation'])['age']
age_group_test = _input0.groupby(['sex', 'passenger_class', 'salutation'])['age']
age_group.median()
age_group.transform('median')
_input1.loc[_input1.age.isnull(), 'age'] = age_group.transform('median')
_input0.loc[_input0.age.isnull(), 'age'] = age_group_test.transform('median')
_input1.age.isnull().sum()
plt.figure(figsize=(12, 5))
sns.histplot(x='age', data=_input1)
plt.title('Total Distribuition and density by Age')
plt.xlabel('Age')
plt.figure(figsize=(12, 5))
sns.histplot(x='age', data=_input1, hue='survived')
plt.title('Distribuition and density by Age and Survival')
plt.xlabel('Age')
plt.figure(figsize=(12, 5))
sns.distplot(x=_input1.age, bins=25)
plt.title('Distribuition and density by Age')
plt.xlabel('Age')
plt.figure(figsize=(12, 5))
g = sns.FacetGrid(_input1, col='survived', size=5)
g = g.map(sns.distplot, 'age')
male_df = _input1[_input1.sex == 'male']
plt.figure(figsize=(12, 5))
g = sns.FacetGrid(male_df, col='survived', size=5)
g = g.map(sns.distplot, 'age')
female_df = _input1[_input1.sex == 'female']
plt.figure(figsize=(12, 5))
g = sns.FacetGrid(female_df, col='survived', size=5)
g = g.map(sns.distplot, 'age')
age_index = _input1[(_input1.sex == 'male') & ((_input1.age >= 20) & (_input1.age <= 40)) | (_input1.sex == 'female') & ((_input1.age >= 18) & (_input1.age <= 40))].index
_input1['age_sur'] = 0
_input1.loc[age_index, 'age_sur'] = 1
_input1[['age_sur', 'survived']]
_input1.groupby('age_sur').survived.value_counts()
_input1['age_sur'] = 0
_input1.loc[age_index, 'age_sur'] = 1
plt.figure(figsize=(12, 5))
sns.countplot('age_sur', data=_input1, hue='survived', palette='hls')
plt.ylabel('Count', fontsize=18)
plt.xlabel('Age Dist', fontsize=18)
plt.title('Age Dist ', fontsize=20)
plt.figure(figsize=(12, 5))
g = sns.FacetGrid(_input1, col='survived', size=5)
g = g.map(sns.distplot, 'age_sur')
print(sorted(_input1.age.unique()))
interval = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150)
cats = list(range(len(interval) - 1))
_input1['age_category'] = pd.cut(_input1.age, interval, labels=cats)
_input1['age_category'].head()
_input0['age_category'] = pd.cut(_input0.age, interval, labels=cats)
_input0['age_category'].head()
_input1.age_category.unique()
plt.figure(figsize=(12, 5))
sns.countplot('age_category', data=_input1, hue='survived', palette='hls')
plt.ylabel('Count', fontsize=18)
plt.xlabel('Age Dist', fontsize=18)
plt.title('Age Dist ', fontsize=20)
male_df = _input1[_input1.sex == 'male']
plt.figure(figsize=(12, 5))
sns.countplot('age_category', data=male_df, hue='survived', palette='hls')
plt.ylabel('Count', fontsize=18)
plt.xlabel('Age Dist for Male', fontsize=18)
plt.title('Age Dist ', fontsize=20)
female_df = _input1[_input1.sex == 'female']
plt.figure(figsize=(12, 5))
sns.countplot('age_category', data=female_df, hue='survived', palette='hls')
plt.ylabel('Count', fontsize=18)
plt.xlabel('Age Dist for Female', fontsize=18)
plt.title('Age Dist ', fontsize=20)
age_index = _input1[(_input1.sex == 'male') & _input1.age_category.isin([0]) | (_input1.sex == 'female') & _input1.age_category.isin([0, 1, 2, 3, 4, 5, 6])].index
age_index_test = _input0[(_input0.sex == 'male') & _input0.age_category.isin([0]) | (_input0.sex == 'female') & _input0.age_category.isin([0, 1, 2, 3, 4, 5, 6])].index
age_index
_input1['age_sur'] = 0
_input1.loc[age_index, 'age_sur'] = 1
_input0['age_sur'] = 0
_input0.loc[age_index_test, 'age_sur'] = 1
plt.figure(figsize=(12, 5))
sns.countplot('age_sur', data=_input1, hue='survived', palette='hls')
plt.ylabel('Count', fontsize=18)
plt.xlabel('Age Dist', fontsize=18)
plt.title('Age Dist ', fontsize=20)
_input1.sibling_spouse.unique()
_input1.groupby('sibling_spouse').survived.value_counts(normalize=True).sort_index()
plt.figure(figsize=(12, 5))
sns.countplot('sibling_spouse', data=_input1, hue='survived')
plt.ylabel('Count', fontsize=18)
plt.xlabel('Sibling Dist', fontsize=18)
plt.title('Sibling Dist ', fontsize=20)
male_df = _input1[_input1.sex == 'male']
plt.figure(figsize=(12, 5))
sns.countplot('sibling_spouse', data=male_df, hue='survived')
plt.ylabel('Count', fontsize=18)
plt.xlabel('Male Sibling Dist', fontsize=18)
plt.title('Male Sibling Dist ', fontsize=20)
female_df = _input1[_input1.sex == 'female']
plt.figure(figsize=(12, 5))
sns.countplot('sibling_spouse', data=female_df, hue='survived')
plt.ylabel('Count', fontsize=18)
plt.xlabel('Female Sibling Dist', fontsize=18)
plt.title('Female Sibling Dist ', fontsize=20)
_input1.parent_children.unique()
_input1.groupby('parent_children').survived.value_counts(normalize=True).sort_index()
plt.figure(figsize=(12, 5))
sns.countplot('parent_children', data=_input1, hue='survived')
plt.ylabel('Count', fontsize=18)
plt.xlabel('parent_children Dist', fontsize=18)
plt.title('parent_children Dist ', fontsize=20)
male_df = _input1[_input1.sex == 'male']
plt.figure(figsize=(12, 5))
sns.countplot('parent_children', data=male_df, hue='survived')
plt.ylabel('Count', fontsize=18)
plt.xlabel('Male parent_children Dist', fontsize=18)
plt.title('Male parent_children Dist ', fontsize=20)
_input1[_input1.sex == 'male'].groupby('parent_children').survived.value_counts(normalize=True).sort_index()
female_df = _input1[_input1.sex == 'female']
plt.figure(figsize=(12, 5))
sns.countplot('parent_children', data=female_df, hue='survived')
plt.ylabel('Count', fontsize=18)
plt.xlabel('Female parent_children Dist', fontsize=18)
plt.title('Female parent_children Dist ', fontsize=20)
ps_ss_sur_index = _input1[(_input1['sex'] == 'female') & (_input1['sibling_spouse'].isin([0, 1, 2, 3]) | _input1['parent_children'].isin([0, 1, 2, 3]))].index
ps_ss_sur_index_test = _input0[(_input0['sex'] == 'female') & (_input0['sibling_spouse'].isin([0, 1, 2, 3]) | _input0['parent_children'].isin([0, 1, 2, 3]))].index
_input1['ps_ss_sur'] = 0
_input1.loc[ps_ss_sur_index, 'ps_ss_sur'] = 1
_input0['ps_ss_sur'] = 0
_input0.loc[ps_ss_sur_index_test, 'ps_ss_sur'] = 1
_input1.ps_ss_sur.corr(_input1.survived)
print(sorted(_input1.fare.unique()))
plt.figure(figsize=(12, 5))
sns.set_theme(style='whitegrid')
sns.boxplot(x='survived', y='fare', data=_input1, palette='Set3')
plt.title('Survived Fare Rate')
_input1.head()
_input1.fare = _input1.fare.fillna(_input1.fare.mean(), inplace=False)
_input0.fare = _input0.fare.fillna(_input0.fare.mean(), inplace=False)
_input1.cabin.isnull().sum()
cabin_null_index = _input1[_input1.cabin.isnull()].index
cabin_null_index_test = _input0[_input0.cabin.isnull()].index
_input1['is_cabin'] = 1
_input1.loc[cabin_null_index, 'is_cabin'] = 0
_input0['is_cabin'] = 1
_input0.loc[cabin_null_index_test, 'is_cabin'] = 0
_input1.is_cabin.corr(_input1.survived)
_input1.embarked.isnull().sum()
_input1.embarked.unique()
_input1.embarked.value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8)).legend()
sns.displot(x=_input1.embarked)
plt.title('Distribuition of embarked values')
_input1.embarked = _input1.embarked.fillna('S', inplace=False)
_input0.embarked = _input0.embarked.fillna('S', inplace=False)
sns.barplot(x='embarked', y='survived', data=_input1)
_input1.head()
_input1.columns
_input1.sex = _input1.sex.replace({'male': 0, 'female': 1}, inplace=False)
_input0.sex = _input0.sex.replace({'male': 0, 'female': 1}, inplace=False)
subset = _input1[['passenger_class', 'survived', 'sal_sur', 'age_sur', 'age_category', 'ps_ss_sur', 'is_cabin', 'sex', 'fare']]
subset_test = _input0[['passenger_class', 'sal_sur', 'age_sur', 'age_category', 'ps_ss_sur', 'is_cabin', 'sex', 'fare']]
subset
colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(subset.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = subset.drop('survived', axis=1)
Y = _input1['survived']
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=10)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')