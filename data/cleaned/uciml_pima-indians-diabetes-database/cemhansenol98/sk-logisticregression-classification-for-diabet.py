import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
import seaborn as sns
sns.pairplot(df.iloc[:, 1:6])
df.info()
df.describe()
y = df.Outcome.values
y
x_data = df.drop(['Outcome'], axis=1)
x_data
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=30)
print('x_train: ', x_train)
print('x_test: ', x_test)
print('y_train: ', y_train)
print('y_test: ', y_test)
print('x_train : ', x_train.shape)
print('x_test : ', x_test.shape)
print('y_train : ', y_train.shape)
print('y_test : ', y_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()