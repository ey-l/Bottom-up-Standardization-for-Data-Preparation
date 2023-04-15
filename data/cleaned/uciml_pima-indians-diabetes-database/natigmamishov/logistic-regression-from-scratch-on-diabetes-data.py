import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import metrics
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
(m, n) = df.shape
x = df.drop('Outcome', axis=1)
Y = df['Outcome']
sc = StandardScaler()
x_scaled = sc.fit_transform(x)
X = x_scaled.tolist()
for row in range(len(X)):
    X[row].insert(0, 1)
print(X[0])
thetas = [0 for i in range(len(X[0]))]
print(thetas)

def hypotesis_function(X, thetas):
    h_x = []
    for row in X:
        summ = 0
        for i in range(len(row)):
            summ += row[i] * thetas[i]
        h_x.append(1 / (1 + np.exp(-summ)))
    return h_x

def log_loss(h_x, Y):
    (m, bernoulli_1, bernoulli_2) = (len(Y), 0, 0)
    for i in range(len(Y)):
        bernoulli_1 += Y[i] * np.log(h_x[i])
        bernoulli_2 += (1 - Y[i]) * np.log(1 - h_x[i])
    LogLoss = -(1 / m) * (bernoulli_1 + bernoulli_2)
    return LogLoss

def gradient_descent(X, Y, h_x):
    gradient = []
    X_trans = np.array(X).T.tolist()
    error = [Y[i] - h_x[i] for i in range(len(Y))]
    for row in X_trans:
        res = 0
        for i in range(len(row)):
            res += row[i] * error[i]
        gradient.append(res)
    return gradient

def update_thetas(thetas, gradient, alpha=0.01):
    new_thetas = []
    for i in range(len(gradient)):
        new_thetas.append(thetas[i] + 1 / m * alpha * gradient[i])
    return new_thetas

def predict(X, w):
    h_x = hypotesis_function(X, thetas)
    return list(map(lambda row: 1 if row >= 0.5 else 0, h_x))

def accuracy_score(Y, Y_pred):
    return (Y == Y_pred).mean()
J_hist = []
for i in range(1000):
    y_pred = hypotesis_function(X, thetas)
    LogLoss = log_loss(y_pred, Y)
    J_hist.append(LogLoss)
    g_d = gradient_descent(X, Y, y_pred)
    thetas = update_thetas(thetas, g_d)
print(thetas)
plt.plot(range(1000), J_hist)
plt.xlabel('Number of Iterarion')
plt.ylabel('Cost')

pred = predict(X, thetas)
print(accuracy_score(Y, pred))

class LogisticRegression:

    def __init__(self, epochs=1000, alpha=0.01):
        self.epochs = epochs
        self.alpha = alpha

    def hypotesis_function(self, X, thetas):
        h_x = []
        for row in X:
            summ = 0
            for i in range(len(row)):
                summ += row[i] * thetas[i]
            h_x.append(1 / (1 + np.exp(-summ)))
        return h_x

    def log_loss(self, h_x, Y):
        (self.m, bernoulli_1, bernoulli_2) = (len(Y), 0, 0)
        for i in range(len(Y)):
            bernoulli_1 += Y[i] * np.log(h_x[i])
            bernoulli_2 += (1 - Y[i]) * np.log(1 - h_x[i])
        LogLoss = -(1 / self.m) * (bernoulli_1 + bernoulli_2)
        return LogLoss

    def gradient_descent(self, X, Y, h_x):
        gradient = []
        X_trans = np.array(X).T.tolist()
        error = [Y[i] - h_x[i] for i in range(len(Y))]
        for row in X_trans:
            res = 0
            for i in range(len(row)):
                res += row[i] * error[i]
            gradient.append(res)
        return gradient

    def update_theta(self, thetas, gradient, alpha=0.01):
        new_thetas = []
        for i in range(len(gradient)):
            new_thetas.append(thetas[i] + 1 / self.m * alpha * gradient[i])
        return new_thetas
    J_hist = []

    def fit(self, X, Y):
        X = X.tolist()
        for row in range(len(X)):
            X[row].insert(0, 1)
        self.thetas = [0 for i in range(len(X[0]))]
        for i in range(1000):
            y_pred = self.hypotesis_function(X, self.thetas)
            LogLoss = self.log_loss(y_pred, Y)
            J_hist.append(LogLoss)
            g_d = self.gradient_descent(X, Y, y_pred)
            self.thetas = self.update_theta(self.thetas, g_d)

    def predict(self, X_test):
        h_x = hypotesis_function(X_test, self.thetas)
        return list(map(lambda row: 1 if row >= 0.5 else 0, h_x))
lr = LogisticRegression()