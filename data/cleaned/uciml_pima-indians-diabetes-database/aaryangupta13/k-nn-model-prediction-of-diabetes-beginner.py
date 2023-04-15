import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')

print(df.info())
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df[col] = df[col].replace(0, np.NaN)
    mean = int(df[col].mean(skipna=True))
    df[col] = df[col].replace(np.NaN, mean)
df
x = df.iloc[:, 0:8]
y = df.iloc[:, 8]
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=123)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
math.sqrt(len(y_test))
classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')