import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from functools import partial

def plotter(data, columns, fig_size):
    """function plots the density for bunch of columns in a data"""
    rows = math.ceil(len(columns) / 4)
    i = 0
    sns.set_style('darkgrid')
    plt.subplots(rows, 4, figsize=fig_size)
    plt.tight_layout()
    for col in columns:
        i += 1
        plt.subplot(rows, 4, i)
        sns.kdeplot(data[col], shade=True)

class Learning_curve:
    """plots the learning curve"""

    def __init__(self, train_x, train_y, val_x, val_y, model):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.model = model

    def learning_curve(self):
        loss_tr = []
        loss_val = []
        points = np.linspace(10, len(self.train_x), 50)
        for i in points:
            i = math.ceil(i)
            score = cross_val_score(self.model, self.train_x[:i], self.train_y[:i], cv=3, scoring='accuracy').mean()