import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
dataset_path = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
data = pd.read_csv(dataset_path)
X = data[data.columns.difference(['Outcome'])]
Y = data['Outcome']
pass
cor = data.corr()
pass
X = X[X.columns.difference(['BloodPressure', 'SkinThickness'])]
X = (X - X.min()) / (X.max() - X.min())
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.3, random_state=0)
tsne = TSNE(n_components=2, learning_rate=10, init='random')
X_train_tsne = tsne.fit_transform(X_train)
pass
X_test_tsne = tsne.fit_transform(X_test)
pass
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=0.0001, random_state=0)
X_test_clus = kmeans.fit_predict(X_test_tsne)
c = ['green' if X_test_clus[i] == 0 else 'blue' for i in X_test_clus]
n_incorrect = 0
for i in range(0, len(c)):
    if type(X_test_clus[0]) == type(list(Y_test)[0]):
        c[i] = 'red'
        n_incorrect += 1
print(f'Accuracy = {(len(Y_test) - n_incorrect) / len(Y_test) * 100}% for {len(Y_test)} samples.')
pass