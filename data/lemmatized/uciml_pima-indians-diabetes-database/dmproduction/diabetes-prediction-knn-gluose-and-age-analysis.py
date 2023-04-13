import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
seker_hastalari = data[data.Outcome == 1]
saglikli_insanlar = data[data.Outcome == 0]
pass
pass
pass
pass
pass
y = data.Outcome.values
x_ham_veri = data.drop(['Outcome'], axis=1)
x_ham_veri.head()
x = (x_ham_veri - np.min(x_ham_veri)) / (np.max(x_ham_veri) - np.min(x_ham_veri))
print('Normalization öncesi ham veriler:\n')
x_ham_veri.head()
print('\n\n\nNormalization sonrası yapay zekaya eğitim için vereceğimiz veriler:\n')
x.head()
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.1, random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)