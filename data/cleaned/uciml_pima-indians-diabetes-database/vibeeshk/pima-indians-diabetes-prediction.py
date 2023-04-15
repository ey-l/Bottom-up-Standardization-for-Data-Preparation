import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
filepath = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
dataset = pd.read_csv(filepath)
dataset
zeronotacc = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for column in zeronotacc:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)
x = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=0, test_size=0.2)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
import math
math.sqrt(len(y_test))
classify = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')