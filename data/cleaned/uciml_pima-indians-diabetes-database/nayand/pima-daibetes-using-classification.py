import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.isnull().sum()
df.info()
sns.relplot(data=df, y='Age', x='Pregnancies', hue='Outcome', kind='line')
sns.relplot(data=df, y='BloodPressure', x='Age', kind='line', hue='Outcome')
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.35, random_state=0)