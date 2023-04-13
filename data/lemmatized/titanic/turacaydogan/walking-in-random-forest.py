import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1.describe()
maleProb = _input1.Survived.loc[_input1.Sex == 'male'].mean()
print('male surviving probability:', maleProb)
femaleProb = _input1.Survived.loc[_input1.Sex == 'female'].mean()
print('female surviving probability:', femaleProb)
sns.catplot(x='Sex', y='Age', hue='Survived', kind='swarm', data=_input1)
sns.catplot(x='Pclass', y='Survived', kind='bar', data=_input1)
sns.catplot(x='Sex', y='Survived', hue='Pclass', kind='bar', data=_input1)
Families = _input1.Survived.loc[_input1.Parch > 0].mean()
print('Parents&Childs:', Families)
Lovers = _input1.Survived.loc[(_input1.Parch == 0) & (_input1.SibSp == 1)].mean()
print('Lovers:', Lovers)
Loners = _input1.Survived.loc[_input1.Parch == 0].mean()
print('Loners:', Loners)
totalLoners = _input1.Survived.loc[(_input1.Parch == 0) & (_input1.SibSp == 0)].mean()
print('Total Loners:', totalLoners)
richLoners = _input1.Survived.loc[(_input1.Parch == 0) & (_input1.SibSp == 0) & (_input1.Pclass == 1)].mean()
print('Rich Loners:', richLoners)
train_x = _input1[['Age', 'Sex', 'Parch', 'Pclass', 'SibSp']]
print(train_x.dtypes)
train_y = _input1[['Survived']]
train_x.Sex.loc[train_x.Sex == 'male'] = 0
train_x.Sex.loc[train_x.Sex == 'female'] = 1
train_x.Sex = train_x['Sex'].astype('str').astype(int)
print(train_x.dtypes)
missingColumns_x = [col for col in train_x.columns if train_x[col].isnull().any()]
print(missingColumns_x)
missingColumns_y = train_y.isnull().any()
print(missingColumns_y)
knn_imputer = KNNImputer(n_neighbors=4, weights='uniform')
train_xi = pd.DataFrame(knn_imputer.fit_transform(train_x))
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input0.describe()
test_x = _input0[['Age', 'Sex', 'Parch', 'Pclass', 'SibSp']]
test_x.Sex.loc[test_x.Sex == 'male'] = 0
test_x.Sex.loc[test_x.Sex == 'female'] = 1
test_x.Sex = test_x['Sex'].astype('str').astype(int)
print(train_x.dtypes)
missingColumns_x = [col for col in test_x.columns if test_x[col].isnull().any()]
print(missingColumns_x)
knn_imputer = KNNImputer(n_neighbors=4, weights='uniform')
test_xi = pd.DataFrame(knn_imputer.fit_transform(test_x))
rfmodel = RandomForestClassifier(random_state=1)