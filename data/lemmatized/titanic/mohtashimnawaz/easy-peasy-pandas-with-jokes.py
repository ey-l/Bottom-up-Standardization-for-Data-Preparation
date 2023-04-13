import pandas as pd
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
print('Pandas - We have read the whole data hooman, within fraction of seconds, you slow snail. Haha..')
print("Me - I'll go for you head Pandas")
_input1.head(3)
print('Me - ...and then for you tail... , I am the mighty Hooman')
_input0.tail()
print('Shape of training data', _input1.shape)
print('Shape of test data', _input0.shape)
print('Train set columns:', _input1.columns)
print('Test set columns:', _input0.columns)
_input1['Name']
_input1.Name
print('Me - I summon the rows from 5 to 10, bring them to me.')
print('Pandas - At your service, Hooman')
_input1[5:11]
_input1['Name'][0:5]
_input1[['Name', 'Age', 'Sex']]
print(type(_input1[['Name']]))
print(type(_input1['Name']))
print('Me - Pandas, show me the magic...')
print('Le Pandas...')
print()
print(_input1.iloc[5])
print(type(_input1.iloc[5]))
print('Type: ', type(_input1.iloc[1:5]))
_input1.iloc[1:5]
_input1.iloc[2, 1]
_input1.loc[2]
print('Me - Pandas...Is it everything?')
print("Pandas - Don't under estimate the power of Pandas, you hooman...")
_input1.describe()
_input1.sort_values('Age')
_input1.sort_values(['Age', 'Name'], ascending=[1, 0])
total = _input1.append(_input0)
print(total.shape)
print('Yes! We have appended the data successfully')
total[total['Pclass'] == 1]
print('Rich kids:')
print()
total[(total['Pclass'] == 1) & (total['Age'] < 16)]
print('Finding count of null values in all columns....')
total.isnull().sum()
total['Pclass'].value_counts()
print(type(total))
total.loc[(total['Pclass'] == 1) & (total['Age'] < 16), 'Pclass'] = 0
print('I hope it does the task :(')
print('New Pclass should be zero')
total[(total['Pclass'] == 0) & (total['Age'] < 16)]
print('See we added the new Pclass that was not before')
print('To confirm: I summon the bonus I gave to you:::)))')
total['Pclass'].value_counts()
total['Family'] = total['Parch'] + total['SibSp']
total.head()
print('Mean of Age:', total['Age'].mean())
print('Median of Age:', total['Age'].median())
print('Minimum Age:', total['Age'].min())
print('Maximum Age:', total['Age'].max())
print('Sum of all age (IDK why I am finding it): ', total['Age'].sum())
print('It gives mean of ages grouped by Pclass:')
total.groupby(['Pclass']).mean()
total = total.drop(['Parch', 'SibSp'], inplace=False, axis=1)
total.head()