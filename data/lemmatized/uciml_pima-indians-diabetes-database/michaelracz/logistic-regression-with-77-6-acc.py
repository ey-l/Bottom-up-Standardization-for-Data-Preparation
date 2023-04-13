import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
file = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
df = pd.read_csv(file)
df.head()
df.info()
df.describe()
pass
pass
pass
pass
pass
pass
pass
corrmat = df.corr()
pass
corrmat
df['Outcome'].value_counts()
testSet = df[['Glucose', 'BMI']]
targetSet = df[['Outcome']]
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(testSet, targetSet, test_size=0.25, random_state=101)
x_train.shape
y_train.shape
from sklearn.linear_model import LogisticRegression
logisticRegre = LogisticRegression()