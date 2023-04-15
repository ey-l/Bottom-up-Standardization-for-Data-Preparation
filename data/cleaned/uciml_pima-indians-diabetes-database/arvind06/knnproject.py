import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset
zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for i in zero:
    dataset[i] = dataset[i].replace(0, np.NaN)
    mean = int(dataset[i].mean(skipna=True))
    dataset[i] = dataset[i].replace(np.NaN, mean)
dataset
x = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=0, test_size=0.2)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
x_train[0]
import math
math.sqrt(len(y_test))
classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')