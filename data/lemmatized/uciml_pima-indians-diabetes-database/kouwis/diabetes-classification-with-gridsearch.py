import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import stdev
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.tail()
data.isnull().any()
data.duplicated().any()
data.info()
dataClean = data.copy()

def RemoveOutliers(df):
    std = stdev(df) * 3
    mean = df.mean()
    limitL = mean - std
    limitR = mean + std
    outliers = dataClean.loc[(df > limitR) | (df < limitL)]
    dataClean.drop(outliers.index, inplace=True)
    return dataClean
dataClean.hist(figsize=(15, 12))
pass
pass
pass
pass
pass
pass
for bar in ax.patches:
    bar_value = bar.get_height()
    text = f'{bar_value:,}'
    text_x = bar.get_x() + bar.get_width() / 2
    text_y = bar.get_y() + bar_value
    bar_color = bar.get_facecolor()
    ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color, size=12)
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
for bar in ax.patches:
    bar_value = bar.get_height()
    text = f'{bar_value:,}'
    text_x = bar.get_x() + bar.get_width() / 2
    text_y = bar.get_y() + bar_value
    bar_color = bar.get_facecolor()
    ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color, size=12)
pass
pass
pass
for bar in ax.patches:
    bar_value = bar.get_height()
    text = f'{bar_value:,}'
    text_x = bar.get_x() + bar.get_width() / 2
    text_y = bar.get_y() + bar_value
    bar_color = bar.get_facecolor()
    ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color, size=12)
pass
pass
pass
pass
pass
pass
corr = dataClean.corr()
pass
matrix = np.triu(corr)
pass
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
features = dataClean.drop(['Outcome'], axis=1)
targets = dataClean.Outcome
(X_train, X_test, y_train, y_test) = train_test_split(features, targets, test_size=0.2, random_state=42)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
res_dfTrain = {}
res_dfTest = {}
GS = {'n_neighbors': np.arange(1, 20), 'weights': ['distance', 'uniform'], 'p': np.arange(1, 5), 'algorithm': ['ball_tree', 'kd_tree', 'auto']}
knn = KNeighborsClassifier()
knn_GS = GridSearchCV(knn, GS, cv=5)