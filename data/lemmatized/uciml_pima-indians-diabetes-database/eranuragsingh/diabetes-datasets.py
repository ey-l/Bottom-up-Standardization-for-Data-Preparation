import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
import seaborn as sns
import matplotlib.pyplot as plt
DF = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
DF
x = DF.iloc[:, [0, 1, 2, 3]].values
x
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)