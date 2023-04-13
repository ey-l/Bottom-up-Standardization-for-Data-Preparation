import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import model_selection
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.head(5)
dataset.info()
dataset.describe()
dataset.isnull().sum()
pass
pass
pass
pass
pass
pass
data = dataset.loc[:, ['BMI', 'Insulin']]
from sklearn.cluster import KMeans
kmeans1 = KMeans(n_clusters=2)