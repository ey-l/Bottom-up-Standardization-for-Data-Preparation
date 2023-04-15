import pandas as pd
diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.info()
diabetes.head()
import seaborn as sns

sns.countplot(x='Outcome', data=diabetes, palette='hls')
diabetes.groupby('Outcome').mean()
import numpy as np
from sklearn import linear_model, datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
diabetes.isnull().sum()
sns.boxplot(x='Outcome', y='Glucose', data=diabetes, palette='hls')
sns.heatmap(diabetes.corr())
(X_train, X_test, y_train, y_test) = train_test_split(diabetes.drop('Outcome', 1), diabetes['Outcome'], test_size=0.3, random_state=25)
LogReg = LogisticRegression()