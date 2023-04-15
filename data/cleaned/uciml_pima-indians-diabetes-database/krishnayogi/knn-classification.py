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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
Dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
Dataset.head()
Dataset.shape
Dataset.info()
Dataset.isnull()
Dataset.isnull().sum()
X = Dataset.drop('Outcome', axis=1).values
y = Dataset['Outcome'].values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for (i, k) in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)