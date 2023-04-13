import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from yellowbrick.regressor import residuals_plot
np.random.seed = 42

def f(x):
    """
    Didatical function
    """
    return x ** 3 - 2 * x ** 2 - 11 * x + 12
lower_bound = -5
upper_bound = 5
n_points = 100
x = np.random.uniform(lower_bound, upper_bound, n_points)
x_train = x[0:80]
y_train = f(x_train)
x_test = x[80:100]
y_test = f(x_test)
plt.figure(figsize=(8, 8))
sns.scatterplot(x=x_train, y=y_train)
plt.legend(['Train points'])

def knn(x_train, y_train, x_0, k=2):
    """
    Simulate the k-NN algorithm 
    
    :params:
    x: x train samples
    
    y: y train samples
    
    x_0: query points
    
    k: Number of nearest  neighbors taken into account
    """
    distances = [np.linalg.norm(x - x_0) for x in x_train]
    result = []
    for (d, y) in zip(distances, y_train):
        result.append((d, y))
    result.sort(key=lambda tup: tup[0])
    result = np.array(result)
    k_results = result[:k, 1]
    return np.mean(k_results)
y_hat = knn(x_train, y_train, -2, k=2)
y_hat
y_real = f(-2)
y_real
y_real - y_hat
y_hat = knn(x_train, y_train, -2, k=3)
y_hat
y_real - y_hat
y_hat = knn(x_train, y_train, -2, k=5)
y_hat
y_real - y_hat
knn(x_train, y_train, -3, k=80)
knn(x_train, y_train, 0, k=80)
knn(x_train, y_train, 2, k=80)
np.mean(y_train)
y_hat = [knn(x_train, y_train, x, k=3) for x in x_test]
plt.figure(figsize=(8, 8))
sns.scatterplot(x=x_test, y=y_test)
sns.scatterplot(x=x_test, y=y_hat)
plt.legend(['Test points', 'Estimated point'])
mean_squared_error(y_test, y_hat, squared=False)
r2_score(y_test, y_hat)

class MyOwnKnnRegression(BaseEstimator):

    def __init__(self, k_neighbors):
        """
        Here we will define the pipeline for each tree.
        
        :params:
        n_estimators: The number of nearest points
        """
        self.k = k_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self.knn(x) for x in X]

    def knn(self, x_0):
        """
        Simulate the k-NN algorithm 

        :params:
        x: x train samples

        y: y train samples

        x_0: query points

        k: Number of nearest  neighbors taken into account
        """
        distances = [np.linalg.norm(x - x_0) for x in self.X_train]
        result = []
        for (d, y) in zip(distances, self.y_train):
            result.append((d, y))
        result.sort(key=lambda tup: tup[0])
        result = np.array(result)
        k_results = result[:self.k, 1]
        return np.mean(k_results)
knn = MyOwnKnnRegression(k_neighbors=3)