import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_1 = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data_1.head()
data_1.shape
data_1.isnull().sum()
data_1.describe()
data_1.query('Glucose > Insulin')
data_1.dtypes
data_1['Outcome'].value_counts()
data_1.corr()
pass
for column in data_1:
    pass
    pass
Y = data_1.iloc[:, 8]
X = data_1.iloc[:, 0:7]
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
x_train.head()
y_train.head()
lm = LinearRegression()