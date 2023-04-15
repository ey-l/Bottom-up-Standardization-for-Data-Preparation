import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
x_data = data.drop(['Outcome'], axis=1)
y = data.Outcome.values
x_data.head()
x_data.tail()
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x.head()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=42)
print('x_train: {}'.format(x_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_train: {}'.format(y_train.shape))
print('y_test: {}'.format(y_test.shape))
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
print('x_train: {}'.format(x_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_train: {}'.format(y_train.shape))
print('y_test: {}'.format(y_test.shape))

def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return (w, b)

def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head
sigmoid(0)

def forward_and_backward_propagation(w, b, x_train, y_train):
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = np.sum(loss) / x_train.shape[1]
    derivative_weight = np.dot(x_train, (y_head - y_train).T) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {'derivative_weight': derivative_weight, 'derivative_bias': derivative_bias}
    return (cost, gradients)

def update(w, b, x_train, y_train, learning_rate, iteration):
    cost_list = []
    cost_list2 = []
    index = []
    for i in range(iteration):
        (cost, gradients) = forward_and_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        w = w - learning_rate * gradients['derivative_weight']
        b = b - learning_rate * gradients['derivative_bias']
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
    parameters = {'weight': w, 'bias': b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')

    return (parameters, gradients, cost_list)

def prediction(w, b, x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)
    y_prediction = np.zeros((1, x_test.shape[1]))
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            y_prediction[0, i] = 0
        else:
            y_prediction[0, i] = 1
    return y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, iteration):
    dimension = x_train.shape[0]
    (w, b) = initialize_weights_and_bias(dimension)
    (parameters, gradients, cost_list) = update(w, b, x_train, y_train, learning_rate, iteration)
    y_prediction_test = prediction(parameters['weight'], parameters['bias'], x_test)
    print('test accuracy: {} %'.format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
logistic_regression(x_train, y_train, x_test, y_test, learning_rate=0.01, iteration=200)
logistic_regression(x_train, y_train, x_test, y_test, learning_rate=2, iteration=200)
logistic_regression(x_train, y_train, x_test, y_test, learning_rate=2, iteration=400)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42, max_iter=150)