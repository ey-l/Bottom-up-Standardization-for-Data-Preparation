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
print(X[0])
thetas = [0 for i in range(len(X[0]))]
b = 0
L2_param = 0
alpha = 0.01
print(thetas)

def hypotesis_function(X, thetas, b):
    h_x = []
    for row in X:
        summ = 0
        for i in range(len(row)):
            summ += row[i] * thetas[i]
        h_x.append(1 / (1 + np.exp(-(summ + b))))
    return h_x

def log_loss(h_x, Y, thetas, L2_param):
    (m, bernoulli_1, bernoulli_2) = (len(Y), 0, 0)
    for i in range(len(Y)):
        bernoulli_1 += Y[i] * np.log(h_x[i])
        bernoulli_2 += (1 - Y[i]) * np.log(1 - h_x[i])
    LogLoss = -(1 / m) + (bernoulli_1 + bernoulli_2)
    LogLoss = LogLoss + L2_param / (2 * m) * sum([i ** 2 for i in thetas])
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
    (d_thetas, new_thetas) = ([], [])
    for i in range(len(gradient)):
        d_thetas.append(-(1 / m) * gradient[i])
    for i in range(len(thetas)):
        new_thetas.append(thetas[i] * (1 - alpha * L2_param / m) - alpha * d_thetas[i])
    return new_thetas

def update_b(b, Y, h_x, alpha):
    error = [Y[i] - h_x[i] for i in range(len(Y))]
    db = -(1 / m) * sum(error)
    b = b - alpha * db
    return b

def predict(X, w, b):
    h_x = hypotesis_function(X, thetas, b)
    return list(map(lambda row: 1 if row >= 0.5 else 0, h_x))

def accuracy_score(Y, Y_pred):
    return (Y == Y_pred).mean()
J_hist = []
for i in range(1000):
    y_pred = hypotesis_function(X, thetas, b)
    LogLoss = log_loss(y_pred, Y, thetas, L2_param)
    J_hist.append(LogLoss)
    g_d = gradient_descent(X, Y, y_pred)
    (thetas, b) = (update_thetas(thetas, g_d), update_b(b, Y, y_pred, alpha))
print(thetas)
print(b)
Y_pred = predict(X, thetas, b)
print(accuracy_score(Y, Y_pred))
pred = predict(X, thetas, b)
print(accuracy_score(Y, pred))

class LogisticRegression:

    def __init__(self, epochs=1000, alpha=0.01, L2_param=0):
        self.epochs = epochs
        self.alpha = alpha
        self.L2_param = L2_param

    def hypotesis_function(self, X, thetas, b):
        h_x = []
        for row in X:
            summ = 0
            for i in range(len(row)):
                summ += row[i] * thetas[i]
            h_x.append(1 / (1 + np.exp(-(summ + b))))
        return h_x

    def log_loss(self, h_x, Y):
        (self.m, bernoulli_1, bernoulli_2) = (len(Y), 0, 0)
        for i in range(len(Y)):
            bernoulli_1 += Y[i] * np.log(h_x[i])
            bernoulli_2 += (1 - Y[i]) * np.log(1 - h_x[i])
        LogLoss = -(1 / self.m) + (bernoulli_1 + bernoulli_2)
        LogLoss = LogLoss + self.L2_param / (2 * self.m) * sum([i ** 2 for i in thetas])
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

    def update_thetas(self, gradient):
        (d_thetas, new_thetas) = ([], [])
        for i in range(len(gradient)):
            d_thetas.append(-(1 / m) * gradient[i])
        for i in range(len(self.thetas)):
            new_thetas.append(self.thetas[i] * (1 - self.alpha * self.L2_param / self.m) - self.alpha * d_thetas[i])
        return new_thetas

    def update_b(self, b, Y, h_x):
        error = [Y[i] - h_x[i] for i in range(len(Y))]
        db = -(1 / m) * sum(error)
        new_b = self.b - self.alpha * db
        return new_b

    def fit(self, X, Y):
        X = X.tolist()
        self.thetas = [0 for i in range(len(X[0]))]
        self.b = 0
        for i in range(1000):
            y_pred = self.hypotesis_function(X, self.thetas, self.b)
            LogLoss = self.log_loss(y_pred, Y)
            g_d = self.gradient_descent(X, Y, y_pred)
            (self.thetas, self.b) = (self.update_thetas(g_d), self.update_b(b, Y, y_pred))

    def predict(self, X_test):
        h_x = hypotesis_function(X_test, self.thetas, self.b)
        return list(map(lambda row: 1 if row >= 0.5 else 0, h_x))
lr = LogisticRegression()