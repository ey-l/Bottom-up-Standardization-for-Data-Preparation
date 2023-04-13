import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.info()
data.describe()
print('Number of People Surveyed is equal to ' + str(len(data)))
data.groupby('Outcome').size()
data.columns
data.drop('Outcome', axis=1).plot(kind='box', subplots=True, layout=(4, 3), sharex=False, sharey=False, figsize=(15, 15))
scatter_matrix(data.drop('Outcome', axis=1), figsize=(12.5, 12.5))
pass
pass
pass
pass
pass
data.hist(figsize=(15, 7.5))
data.isnull()
data.isnull().sum()
pass
pass
pass
from sklearn.model_selection import train_test_split
X = data.drop('Outcome', axis=1)
y = data['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=1, test_size=0.3)
len(X_train)
len(X_test)
X_test.info()
X_train.info()
categorical = [var for var in data.columns if data[var].dtype == 'O']
print('There are {} categorical variables\n'.format(len(categorical)))
numerical = [var for var in data.columns if data[var].dtype != 'O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)
(X_train.shape, X_test.shape)
X_train.dtypes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()