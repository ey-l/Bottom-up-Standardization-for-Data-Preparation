import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pass
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.info()
df.describe()
df.isnull().sum()
df.info()
pass
pass
df.shape
df
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def models(X_train, Y_train):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)