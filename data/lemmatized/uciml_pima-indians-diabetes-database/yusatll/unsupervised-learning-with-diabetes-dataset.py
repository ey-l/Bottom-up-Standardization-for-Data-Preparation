import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head(10)
data.info()
data.describe()
pass
pass
pass
pass
pass
pass
data1 = data.loc[:, ['Glucose', 'Insulin']]
from sklearn.cluster import KMeans
kmeans1 = KMeans(n_clusters=2)