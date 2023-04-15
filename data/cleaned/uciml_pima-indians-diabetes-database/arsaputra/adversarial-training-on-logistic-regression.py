import pandas as pd
import numpy as np
from numpy import log, dot, e, ndim
from numpy.random import rand
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
features = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
output = data[['Outcome']]
X = features.to_numpy()
y = output.to_numpy()
y[y == [0]] = [-1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y, test_size=0.2)
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y, test_size=0.2)

class LogisticRegression:

    def sigmoid(self, z):
        return 1 / (1 + e ** (-z))

    def grad_vector(self, X, y, index):
        X_i = X[index]
        y_vect = np.array([y[index][0] for i in range(len(X_i))])
        XY = -X_i * y_vect
        return np.append(XY, -y[index][0])

    def multiplier(self, X, y, index, weights, b):
        numerator = e ** (-y[index][0] * (dot(weights, X[index]) + b))
        denumerator = 1 + numerator
        return numerator / denumerator

    def gradient(self, X, y, weights, b):
        vector = np.zeros(len(weights) + 1)
        for i in range(len(X)):
            vector += self.multiplier(X, y, i, weights, b) * self.grad_vector(X, y, i)
        return vector

    def cost_function(self, X, y, weights, b):
        cost = 0
        for i in range(len(X)):
            cost += log(1 + e ** (-y[i][0] * (dot(weights, X[i]) + b)))
        return cost

    def fit(self, X, y, epochs=500, alpha=0.0001, adv=False):
        if adv == False:
            training_cost = []
            weights_and_bias = rand(len(X[0]) + 1)
            self.init_wb = weights_and_bias
            for i in range(epochs):
                weights = weights_and_bias[:8]
                bias = weights_and_bias[-1]
                weights_and_bias -= alpha * self.gradient(X, y, weights, bias)
                weights = weights_and_bias[:8]
                bias = weights_and_bias[-1]
                cost = self.cost_function(X, y, weights, bias)
                training_cost.append(cost)
            self.weights = weights
            self.bias = bias
            self.training_cost = training_cost
        else:
            training_cost = []
            weights_and_bias = self.init_wb
            weights = weights_and_bias[:8]
            bias = weights_and_bias[-1]
            X_og = X
            for i in range(epochs):
                X_adv = self.adversarial_examples(X_og, weights)
                weights_and_bias -= 0.5 * alpha * (self.gradient(X_og, y, weights, bias) + self.gradient(X_adv, y, weights, bias))
                weights = weights_and_bias[:8]
                bias = weights_and_bias[-1]
                cost = self.cost_function(X, y, weights, bias)
                training_cost.append(cost)
            self.weights = weights
            self.bias = bias
            self.training_cost_adv = training_cost

    def predict(self, X):
        z = dot(X, self.weights) + self.bias
        return [[1 if i >= 0.5 else -1 for i in self.sigmoid(z)], [i if i >= 0.5 else 1 - i for i in self.sigmoid(z)]]

    def adversarial_examples(self, X, weights, eps=0.0001):
        return np.array([x - eps * np.sign(weights) for x in X])
logreg = LogisticRegression()