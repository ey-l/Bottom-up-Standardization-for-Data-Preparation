import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(2)
import seaborn as sns
pass
x = df.iloc[:, :8]
y = df.iloc[:, 8]
from sklearn import preprocessing
x = preprocessing.scale(x)
x
pass
from sklearn.linear_model import LogisticRegression
logit_reg = LogisticRegression()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)
x_train
y_train