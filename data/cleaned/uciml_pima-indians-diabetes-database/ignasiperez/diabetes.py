import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(15)
df.value_counts()
sns.pairplot(df, hue='Outcome')
y = df['Outcome']
X = df.drop(['Outcome'], axis=1)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier()