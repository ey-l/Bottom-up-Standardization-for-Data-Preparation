import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print('Dataset :', data.shape)
data.info()
data[0:10]
data.Outcome.value_counts()[0:30].plot(kind='bar')

sns.set_style('whitegrid')
sns.pairplot(data, hue='Outcome', size=3)

sns.boxplot(x='Outcome', y='Age', data=data)

from sklearn.model_selection import train_test_split
Y = data['Outcome']
X = data.drop(columns=['Outcome'])
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.1, random_state=9)
print('X train shape: ', X_train.shape)
print('Y train shape: ', Y_train.shape)
print('X test shape: ', X_test.shape)
print('Y test shape: ', Y_test.shape)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=10)