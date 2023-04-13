import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
_input1.shape
_input1.dtypes
_input1.info()
_input1.columns
_input1.head()
_input1.tail()
plt.figure(figsize=(4, 4))
plt.bar(list(_input1['Survived'].value_counts().keys()), list(_input1['Survived'].value_counts()), color=('grey', 'silver'))
plt.title('Num of Survived People')
_input1['Survived'].value_counts()
sum(_input1['Survived'].isnull())
plt.figure(figsize=(5, 5))
plt.bar(list(_input1['Pclass'].value_counts().keys()), list(_input1['Pclass'].value_counts()))
plt.title('Passengers CLass')
plt.xlabel('Classes')
plt.ylabel('Num of Passengers')
_input1['Pclass'].value_counts()
plt.figure(figsize=(5, 5))
plt.bar(list(_input1['Sex'].value_counts().keys()), list(_input1['Sex'].value_counts()))
plt.title('Gender')
plt.ylabel('Num of Passengers')
_input1['Sex'].value_counts()
plt.figure(figsize=(5, 5))
plt.hist(_input1['Age'])
plt.title('AGE')
plt.figure(figsize=(5, 5))
plt.bar(list(_input1['Age'].value_counts().keys()), list(_input1['Age'].value_counts()))
plt.title('Age')
plt.ylabel('Num of Passengers')
sns.countplot(x='Survived', hue='Sex', data=_input1, palette='winter')
sns.boxplot(x='Pclass', y='Age', data=_input1)
sns.boxplot(x='Sex', y='Age', data=_input1)
_input1.isnull().sum()
sum(_input1['Age'].isnull())
pd.get_dummies(_input1['Sex'])
train_file_update = _input1.dropna()
plt.scatter(train_file_update['Age'], train_file_update['Survived'], marker='*')
train_file_update = _input1.dropna()
print(sum(train_file_update['Survived'].isnull()))
print(sum(train_file_update['Age'].isnull()))
x_train_data = train_file_update['Age']
y_train_data = train_file_update['Survived']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(train_file_update[['Age']], train_file_update.Survived, test_size=0.1)
print(sum(y_test), sum(y_train))
x_train
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()