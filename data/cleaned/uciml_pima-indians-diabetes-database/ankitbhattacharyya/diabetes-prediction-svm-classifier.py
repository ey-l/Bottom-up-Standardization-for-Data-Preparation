import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
diabetes_dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_dataset.head()
diabetes_dataset.tail()
diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
X = diabetes_dataset.drop(columns=['Outcome'], axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)
scaler = StandardScaler()