import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
input_diabetes_data_1 = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
input_diabetes_data_1.head()
input_diabetes_data_1.shape
input_diabetes_data_2 = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
frames = [input_diabetes_data_1, input_diabetes_data_2]
input_diabetes_data = pd.concat(frames)
input_diabetes_data.shape
input_diabetes_data.describe().T
input_diabetes_data = input_diabetes_data.drop_duplicates()
input_diabetes_data.describe().T
np.sum(input_diabetes_data.isnull())
input_diabetes_data.shape
input_diabetes_data.hist(figsize=(25, 25), color='Green', edgecolor='black')
Non_Zero_Columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in Non_Zero_Columns:
    input_diabetes_data[col] = input_diabetes_data[col].replace(0, np.NaN)
np.sum(input_diabetes_data.isnull()) / input_diabetes_data.shape[0] * 100
Non_Zero_Columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in Non_Zero_Columns:
    mean_col = input_diabetes_data[col].mean(skipna=True)
    input_diabetes_data[col] = input_diabetes_data[col].replace(np.NaN, mean_col)
input_diabetes_data.hist(figsize=(25, 25), color='green', edgecolor='black')
input_diabetes_data.describe().T
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X = input_diabetes_data.iloc[:, 0:8]
y = input_diabetes_data.iloc[:, 8]
print(y)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=1 / 3, random_state=0, stratify=y)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
import math
print(math.sqrt(input_diabetes_data.shape[0]))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(27)