import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
LEN1 = len(_input1)
print('Shape of train data: ', _input1.shape)
_input1.head()
_input1.info()
numerical_columns = _input1.select_dtypes(include=np.number).columns
categorical_columns = _input1.select_dtypes('object').columns
bool_columns = _input1.select_dtypes('bool').columns
print(numerical_columns)
print(len(numerical_columns))
print(categorical_columns)
print(len(categorical_columns))
print(bool_columns)
print(len(bool_columns))
_input1['CryoSleep'].tail(10)
num_nans = _input1.isna().sum().sort_values(ascending=False)
nans = pd.DataFrame({'Number_of_NaNs': num_nans})
nans['%_of_Nans'] = nans['Number_of_NaNs'] * 100 / LEN1
nans
_input1.describe()
_input1.describe(exclude=np.number).drop(['PassengerId', 'Name'], axis=1)
quartile1 = _input1['Age'].quantile(0.25)
quartile2 = _input1['Age'].quantile(0.5)
quartile3 = _input1['Age'].quantile(0.75)
interquartile_range = quartile3 - quartile1
xrange = 1.5 * interquartile_range
border2 = quartile3 + xrange
border1 = quartile1 - xrange
whisker_end_1 = _input1.query('Age <= @border2')['Age'].max()
whisker_end_2 = _input1.query('Age >= @border1')['Age'].min()
plt.figure(figsize=(18, 5))
plt.grid(color='rosybrown', linestyle='-', linewidth=0.5)
plt.axvline(x=border1, color='peru', label='Left End of 1.5 of interquartile range')
plt.axvline(x=whisker_end_2, color='orange', label='End of Left Whisker (minimal non-outlier)')
plt.axvline(x=quartile1, color='limegreen', label='1st quartile')
plt.axvline(x=quartile2, color='black', label='Median (2nd quartile)')
plt.axvline(x=quartile3, color='darkmagenta', label='3rd quartile')
plt.axvline(x=whisker_end_1, color='g', label='End of Right Whisker (maximal non-outlier)')
plt.axvline(x=border2, color='darkorange', label='Right End of 1.5 of interquartile range')
sns.boxplot(x=_input1['Age'], color='deepskyblue')
plt.legend(loc=3, prop={'size': 10})
len(_input1[_input1['Age'] > border2])
plt.figure(figsize=(12, 7))
sns.kdeplot(_input1['Age'], fill=True)
columns = numerical_columns.drop('Age')
(fig, ax) = plt.subplots(3, 2, figsize=(18, 10))
for i in range(3):
    for j in range(2):
        if 2 * i + j != 5:
            sns.kdeplot(data=_input1[_input1.select_dtypes(exclude='object').columns.to_list()], x=columns[2 * i + j], ax=ax[i, j], fill=True)
fig.delaxes(ax[2, 1])
columns
skew_kurt = pd.DataFrame({'Skewness': _input1[numerical_columns].skew(), 'Kurtosis': _input1[numerical_columns].kurt()})
skew_kurt