import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print('Shape of the dataset: ' + str(data.shape))
print('\n \nThe first few rows of the dataset: ')
data.head()
n_missing = data.isnull().sum().sum()
if n_missing == 0:
    print('There  is no missing values in the dataset')
else:
    print('Oops! Total ' + str(n_missing) + ' values in the dataset')
dataset = data.copy()
for col in dataset.columns[0:-1]:
    dataset[col] = dataset[col] / abs(dataset[col].max())
dataset.head()
(train, test) = train_test_split(dataset, test_size=0.3, random_state=42, shuffle=True)

def makeinput(df_in):
    df = df_in.copy()
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = y.reshape(y.shape[0], 1)
    x0 = np.ones(x.shape[0]).reshape(x.shape[0], 1)
    x = np.append(x0, x, axis=1)
    return (x, y)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
a = np.linspace(-5, 5, 200)
b = sigmoid(a)
pass
pass
pass

def hyp(theta, x):
    return np.matmul(x, theta)

def cost(theta, x, y):
    m = x.shape[0]
    h = hyp(theta, x)
    J = -1 / m * np.sum(y * np.log(sigmoid(h)) + (1 - y) * np.log(1 - sigmoid(h)))
    return J

def optim(theta, x, y, alpha, epochs):
    m = x.shape[0]
    j = np.zeros(epochs)
    for i in range(epochs):
        h = hyp(theta, x)
        gd = 1 / m * np.matmul(np.transpose(x), sigmoid(h) - y)
        theta = theta - alpha * gd
        j[i] = cost(theta, x, y)
    return (theta, j)

def pred(theta, x):
    h = hyp(theta, x)
    return sigmoid(h)
(x_train, y_train) = makeinput(train)
(m, n) = x_train.shape
theta_init = np.zeros((n, 1))
loss = cost(theta_init, x_train, y_train)
print('Loss without optimization is ' + str(loss))
alpha = 0.01
epochs = 50000
(theta, j) = optim(theta_init, x_train, y_train, alpha, epochs)
loss_opt = cost(theta, x_train, y_train)
print('Loss after optimization is ' + str(loss_opt))
iteration = range(epochs)
pass
pass
pass
(x_test, y_test) = makeinput(test)
predictions = np.round(pred(theta, x_test))
print(accuracy_score(y_test, predictions))