import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.describe()
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
neighbors = np.arange(1, 9)
dt_train_akurasi = np.empty(len(neighbors))
dt_test_akurasi = np.empty(len(neighbors))
for (i, k) in enumerate(neighbors):
    data_knn = KNeighborsClassifier(n_neighbors=k)