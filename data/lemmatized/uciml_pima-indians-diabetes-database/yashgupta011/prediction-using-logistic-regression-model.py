import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.shape
df.head()
df.isnull().values.any()
columns = list(df)[0:-1]
df[columns].hist(stacked=False, bins=100, figsize=(12, 30), layout=(14, 2))
df.corr()

def plot_corr(df, size=11):
    corr = df.corr()
    pass
    ax.matshow(corr)
    pass
    pass
plot_corr(df)
pass
n_true = len(df.loc[df['Outcome'] == True])
n_false = len(df.loc[df['Outcome'] == False])
print('Number of true cases: {0} ({1:2.2f}%)'.format(n_true, n_true / (n_true + n_false) * 100))
print('Number of false cases: {0} ({1:2.2f}%)'.format(n_false, n_false / (n_true + n_false) * 100))
from sklearn.model_selection import train_test_split
X = df.drop('Outcome', axis=1)
Y = df['Outcome']
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.3, random_state=1)
x_train.head()
print('{0:0.2f}% data is in training set'.format(len(x_train) / len(df.index) * 100))
print('{0:0.2f}% data is in test set'.format(len(x_test) / len(df.index) * 100))
print('Original Diabetes True Values    : {0} ({1:0.2f}%)'.format(len(df.loc[df['Outcome'] == 1]), len(df.loc[df['Outcome'] == 1]) / len(df.index) * 100))
print('Original Diabetes False Values   : {0} ({1:0.2f}%)'.format(len(df.loc[df['Outcome'] == 0]), len(df.loc[df['Outcome'] == 0]) / len(df.index) * 100))
print('')
print('Training Diabetes True Values    : {0} ({1:0.2f}%)'.format(len(y_train[y_train[:] == 1]), len(y_train[y_train[:] == 1]) / len(y_train) * 100))
print('Training Diabetes False Values   : {0} ({1:0.2f}%)'.format(len(y_train[y_train[:] == 0]), len(y_train[y_train[:] == 0]) / len(y_train) * 100))
print('')
print('Test Diabetes True Values        : {0} ({1:0.2f}%)'.format(len(y_test[y_test[:] == 1]), len(y_test[y_test[:] == 1]) / len(y_test) * 100))
print('Test Diabetes False Values       : {0} ({1:0.2f}%)'.format(len(y_test[y_test[:] == 0]), len(y_test[y_test[:] == 0]) / len(y_test) * 100))
print('')
x_train.head()
from sklearn.impute import SimpleImputer
rep_0 = SimpleImputer(missing_values=0, strategy='mean')
cols = x_train.columns
x_train = pd.DataFrame(rep_0.fit_transform(x_train))
x_test = pd.DataFrame(rep_0.fit_transform(x_test))
x_train.columns = cols
x_test.columns = cols
x_train.head()
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')