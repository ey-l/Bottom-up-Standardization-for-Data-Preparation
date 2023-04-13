import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.corr()
import seaborn as sns
pass
corr = df.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i + 1, corr.shape[0]):
        if corr.iloc[i, j] >= 0.8:
            if columns[j]:
                columns[j] = Falseselected_columns = df.columns[columns] = df[selected_columns]
df.head()
y = df.Outcome.values
X = df.drop(columns=['Outcome'], axis=1).values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=40, test_size=0.2)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(X_train)
X_train.shape
y.shape
from sklearn.svm import SVC
model = SVC(kernel='linear', C=10)