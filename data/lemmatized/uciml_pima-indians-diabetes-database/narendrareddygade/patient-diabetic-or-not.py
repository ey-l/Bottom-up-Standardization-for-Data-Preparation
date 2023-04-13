import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info()
data.isnull().sum()
data.nunique().sort_values()
data.describe()
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
from sklearn.model_selection import train_test_split
(xtrain, xtest, ytrain, ytest) = train_test_split(X, Y, test_size=1 / 3, random_state=1)
from sklearn.linear_model import LinearRegression
ls = LinearRegression()