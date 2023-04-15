import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', sep=',')
df.head()
df.info()
correlation = df.corr()
correlation
sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns)
willScale = df.columns.values.tolist()
minMaxScaler = MinMaxScaler()
scaledColums = pd.DataFrame(minMaxScaler.fit_transform(df[willScale]), columns=willScale)
scaledColums.describe()
df = pd.concat([scaledColums], axis=1)
df
target = ['Outcome']
features = df.columns.drop(target)
(train, test) = train_test_split(df, test_size=0.22, random_state=12)
xTrain = train[features]
yTrain = train[target]
xTest = test[features]
yTest = test[target]
dTree = DecisionTreeClassifier(criterion='gini', max_depth=4)
start = time.time()