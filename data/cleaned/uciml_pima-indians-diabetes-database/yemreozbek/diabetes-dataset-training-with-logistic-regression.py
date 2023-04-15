import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(10)
df.info()
y = df.Outcome.values
x_df = df.drop(['Outcome'], axis=1)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x_df, y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()