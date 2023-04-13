import numpy as np
import matplotlib.pyplot as plt
X_train = np.linspace(0, 10 * np.pi, num=1000)
noise = np.random.normal(scale=3, size=X_train.size)
y_train = X_train + 10 * np.sin(X_train) + noise
plt.scatter(X_train, y_train, color='g')
plt.xlabel('training feature')
plt.ylabel('training response')
X_test = np.linspace(10, 15 * np.pi, num=100)
noise = np.random.normal(scale=3, size=X_test.size)
y_true = X_test + 10 * np.sin(X_test) + noise
plt.scatter(X_test, y_true, color='g')
plt.xlabel('test feature')
plt.ylabel('test response')
from sklearn.linear_model import LinearRegression
model = LinearRegression()