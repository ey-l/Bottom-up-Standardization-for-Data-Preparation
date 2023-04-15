import numpy as np
import pandas as pd
import csv
import os
os.getcwd()
diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.head()
diabetes.describe()
diabetes.info()
diabetes['Outcome'].value_counts()
diabetes[diabetes['Outcome'] == 1]['Insulin'].max()
diabetes[diabetes['Outcome'] == 1]['Glucose'].max()
diabetes['Pregnancies'].nunique()
diabetes[diabetes['Outcome'] == 1]['Pregnancies'].value_counts()
diabetes[diabetes['Outcome'] == 0]['Pregnancies'].value_counts()

def age_count(x):
    if x in diabetes[diabetes['Outcome'] == 0]['Age']:
        return False
    else:
        return x
diabetes[diabetes['Outcome'] == 1]['Age'].apply(lambda x: age_count(x)).value_counts()
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(diabetes, hue='Outcome')
sns.heatmap(diabetes.corr(), cmap='coolwarm')
sns.jointplot(data=diabetes, x='Glucose', y='Insulin', kind='hex')
sns.set_style('whitegrid')
sns.countplot(x='Pregnancies', hue='Outcome', data=diabetes, palette='RdBu_r')
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
y = diabetes['Outcome']
x = diabetes.drop('Outcome', axis=1)
(X_train, X_test, Y_train, Y_test) = train_test_split(x, y, test_size=0.3, random_state=101)
logmodel = LogisticRegression()