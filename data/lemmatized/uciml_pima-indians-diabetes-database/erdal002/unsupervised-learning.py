import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', sep=',')
df.head()
pass
df.isnull().sum()
data = df.drop('Outcome', axis=1)
data.head()
wcss = []
for K in range(1, 15):
    Kmeans = KMeans(n_clusters=K)