import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
diab = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diab.head()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
diab.columns.tolist()
diab.isna().sum()
X = diab.drop(labels=['Outcome'], axis=1)
y = diab.Outcome
correlation = diab.corr()
correlation.style.background_gradient(cmap='plasma')
pass
pass
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=10, max_iter=1000)