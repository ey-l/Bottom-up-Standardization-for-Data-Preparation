import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
train.head()
train.isnull().sum()
train.info()
pass
pass
pass
pass
pass
pass
pass
pass
train[train['BMI'] <= 10].count()
train = train[train['BMI'] > 10]
train.shape
pass
pass
train[train['BloodPressure'] < 40].count()
train = train[train['BloodPressure'] > 40]
train.shape
pass
pass
train[train['Glucose'] <= 0].count()
train = train[train['Glucose'] > 0]
train.shape
pass
pass
train[train['SkinThickness'] <= 0].count()
train = train[train['SkinThickness'] > 0]
train.shape
pass
pass
train[train['Insulin'] < 0].count()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(train.drop('Outcome', axis=1), train['Outcome'], test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)