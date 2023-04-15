import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
housing_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
(fig, ax) = plt.subplots(1, 2, figsize=(18, 8), squeeze=False)
sns.scatterplot(data=housing_data, x='GrLivArea', y='SalePrice', ax=ax[0][0])
sns.scatterplot(data=housing_data, x='ScreenPorch', y='SalePrice', ax=ax[0][1])


def evaluate(y_predict, y_expected):
    sq_error = (np.log(y_expected) - np.log(y_predict)) ** 2
    return np.sqrt(np.mean(sq_error))
X = housing_data.GrLivArea.to_numpy()
y = housing_data.SalePrice.to_numpy()
X_b = np.column_stack((np.ones(X.shape[0]), X))
my_theta = np.array([1500, 100])
simple_pred = np.sum(X_b * my_theta, axis=1)
score = evaluate(simple_pred, y)
print('Score:', score)
plt.figure(figsize=(10, 8))
plt.plot(X, y, 'b.', alpha=0.4)
plt.plot(X, simple_pred, 'r-', label='Wild Guess Prediction')
plt.legend(loc='upper right')


def scale_values(X):
    mu = np.mean(X)
    sigma = np.std(X)
    return (X - mu) / sigma

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

def compute_gradient(X, y, w, b):
    (m, n) = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return (dj_db, dj_dw)

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        (dj_db, dj_dw) = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return (w, b)
cols = housing_data.corr().SalePrice.sort_values(ascending=False)[1:11].index
housing_data[cols]
y = housing_data['SalePrice'].to_numpy()
X = housing_data[cols].to_numpy()
X = scale_values(X)
initial_weights = np.zeros(X.shape[1])
initial_b = 0
print('y shape:', y.shape)
print('X shape:', X.shape)
print('initial_weights shape:', initial_weights.shape)
alpha = 0.05
nbr_iters = 2000
(theta, b) = gradient_descent(X, y, initial_weights, initial_b, compute_cost, compute_gradient, alpha, nbr_iters)
print('theta:', theta, 'b:', b)
y_predictions = np.sum(X * theta + b, axis=1)
print('Score:', evaluate(y_predictions, y))
plt.figure(figsize=(10, 8))
plt.plot(X, y, 'b.', alpha=0.4)
plt.plot(X, y_predictions, 'rx', label='Prediction')

test_housing_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
X_test = test_housing_data[cols].to_numpy()
X_test = np.nan_to_num(X_test)
X_test = scale_values(X_test)
y_test = np.sum(X_test * theta + b, axis=1)
predictions_df = pd.DataFrame({'Id': test_housing_data['Id'].values, 'SalePrice': y_test})
