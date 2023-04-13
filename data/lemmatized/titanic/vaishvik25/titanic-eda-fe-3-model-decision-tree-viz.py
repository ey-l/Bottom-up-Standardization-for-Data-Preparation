import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import seaborn as sns
sns.set(style='dark')
sns.set(style='darkgrid', color_codes=True)
RED = '\x1b[1;31m'
BLUE = '\x1b[1;34m'
CYAN = '\x1b[1;36m'
GREEN = '\x1b[0;32m'
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head(5)
_input0.head(5)
a = sum(pd.isnull(_input1['Age']))
b = round(a / len(_input1['PassengerId']), 4)
sys.stdout.write(GREEN)
print('Count of missing Values : {} , The Proportion of this values with dataset is {}\n'.format(a, b * 100))
sys.stdout.write(CYAN)
print('visualization AGE')
ax = _input1['Age'].hist(bins=15, color='#34495e', alpha=0.9)
ax.set(xlabel='Age', ylabel='Count')
m1 = _input1['Age'].median(skipna=True)
m2 = _input1['Age'].mean(skipna=True)
sys.stdout.write(CYAN)
print('Median: {} and Mean: {} | Median age is 28 as compared to mean which is ~30'.format(m1, m2))
a = round(2 / len(_input1['PassengerId']), 4)
sys.stdout.write(CYAN)
print('proportion of "Embarked" missing is {}'.format(a * 100))
sys.stdout.write(CYAN)
print('visualization Embarked')
sns.countplot(x='Embarked', data=_input1, palette='Set1')
train_data = _input1
train_data['Age'] = train_data['Age'].fillna(28, inplace=False)
train_data['Embarked'] = train_data['Embarked'].fillna('S', inplace=False)
train_data = train_data.drop('Cabin', axis=1, inplace=False)
train_data['TravelBuds'] = train_data['SibSp'] + train_data['Parch']
train_data['TravelAlone'] = np.where(train_data['TravelBuds'] > 0, 0, 1)
train_data = train_data.drop('SibSp', axis=1, inplace=False)
train_data = train_data.drop('Parch', axis=1, inplace=False)
train_data = train_data.drop('TravelBuds', axis=1, inplace=False)
train2 = pd.get_dummies(train_data, columns=['Pclass'])
train3 = pd.get_dummies(train2, columns=['Embarked'])
train4 = pd.get_dummies(train3, columns=['Sex'])
train4 = train4.drop('Sex_female', axis=1, inplace=False)
train4 = train4.drop('PassengerId', axis=1, inplace=False)
train4 = train4.drop('Name', axis=1, inplace=False)
train4 = train4.drop('Ticket', axis=1, inplace=False)
train4.head(5)
df_final = train4
_input0['Age'] = _input0['Age'].fillna(28, inplace=False)
_input0['Fare'] = _input0['Fare'].fillna(14.45, inplace=False)
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
_input0['TravelBuds'] = _input0['SibSp'] + _input0['Parch']
_input0['TravelAlone'] = np.where(_input0['TravelBuds'] > 0, 0, 1)
_input0 = _input0.drop('SibSp', axis=1, inplace=False)
_input0 = _input0.drop('Parch', axis=1, inplace=False)
_input0 = _input0.drop('TravelBuds', axis=1, inplace=False)
test2 = pd.get_dummies(_input0, columns=['Pclass'])
test3 = pd.get_dummies(test2, columns=['Embarked'])
test4 = pd.get_dummies(test3, columns=['Sex'])
test4 = test4.drop('Sex_female', axis=1, inplace=False)
test4 = test4.drop('PassengerId', axis=1, inplace=False)
test4 = test4.drop('Name', axis=1, inplace=False)
test4 = test4.drop('Ticket', axis=1, inplace=False)
final_test = test4
final_test.head(5)
sys.stdout.write(GREEN)
print('Density Plot of Age for Surviving Population and Deceased Population')
plt.figure(figsize=(15, 8))
sns.kdeplot(_input1['Age'][df_final.Survived == 1], color='darkturquoise', shade=True)
sns.kdeplot(_input1['Age'][df_final.Survived == 0], color='lightcoral', shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
plt.figure(figsize=(25, 8))
avg_survival_byage = df_final[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color='LightSeaGreen')
df_final['IsMinor'] = np.where(train_data['Age'] <= 16, 1, 0)
final_test['IsMinor'] = np.where(final_test['Age'] <= 16, 1, 0)
plt.figure(figsize=(15, 8))
sns.kdeplot(df_final['Fare'][_input1.Survived == 1], color='#e74c3c', shade=True)
sns.kdeplot(df_final['Fare'][_input1.Survived == 0], color='#3498db', shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
plt.xlim(-20, 200)
sns.barplot('Pclass', 'Survived', data=_input1, color='#2ecc71')
sns.barplot('Embarked', 'Survived', data=_input1, color='#2ecc71')
sns.barplot('TravelAlone', 'Survived', data=df_final, color='#2ecc71')
cols = ['Age', 'Fare', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 'Embarked_S', 'Sex_male', 'IsMinor']
X = df_final[cols]
Y = df_final['Survived']
import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
logit_model = sm.Logit(Y, X)