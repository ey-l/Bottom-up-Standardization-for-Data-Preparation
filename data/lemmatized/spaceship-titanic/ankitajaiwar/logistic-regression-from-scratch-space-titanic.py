import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.describe()
_input1.shape
_input1.head()
_input0.head()
_input1.info()
_input1['Destination'].value_counts()
_input1['Cabin'].value_counts()
_input1 = _input1.drop(['Cabin', 'Name'], axis=1)
_input0 = _input0.drop(['Cabin', 'Name'], axis=1)
_input1 = _input1.dropna()
_input1.info()
_input0 = _input0.dropna()
object_columns = _input1.select_dtypes(include='object').columns
object_columns
le_list = []
for i in object_columns[1:]:
    le = LabelEncoder()
    print(i)
    _input1[i] = le.fit_transform(_input1[i])
    _input0[i] = le.transform(_input0[i])
    le_list.append(le)
ohe_label = LabelEncoder()
_input1['Transported'] = ohe_label.fit_transform(_input1['Transported'])
_input1.head()
_input0.head()
log_likelihood = []

class LogisticRegression:

    def __init__(self, data, labels, lr, max_iteration, stop_dif):
        self.lr = lr
        self.max_iteration = max_iteration
        self.stop_dif = stop_dif
        print(data.shape)
        print(labels.shape)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def prediction(self, X, w):
        """
        This function calculates the results of the sigmoid function which will later be used for predictions.
        """
        return self.sigmoid(np.dot(X, w))

    def log_like(self, data, labels, w, z, prob):
        """
        This function calculates the log likelihood in each iteration. The higher the value of the log-likelihood, the better a model fits a dataset. 
        """
        ll = np.sum(labels * np.log(prob) + (1 - labels) * np.log(1 - prob))
        log_likelihood.append(ll)

    def predict(self, X, w):
        """
        This function predicts the labels.
        """
        pred_labels = np.zeros(X.shape[0])
        preds = self.prediction(X, w) >= 0.5
        for i in range(0, len(preds)):
            if preds[i] == True:
                pred_labels[i] = 1
            else:
                pred_labels[i] = 0
        return pred_labels

    def accuracy(self, X, y_labels, w):
        """
        This function returns the accuracy.
        """
        return (self.predict(X, w) == y_labels).mean()

    def log_reg(self, data, labels):
        """
        This function updates the weights associated with features in each iteration.
        """
        w = np.zeros(data.shape[1])
        w_zero = w
        for step in range(self.max_iteration):
            lambda1 = 0.001
            z = np.dot(data, w)
            prob = self.prediction(data, w)
            gradient = np.dot(data.T, labels - np.exp(z) / (1 + np.exp(z)))
            w = w - self.lr * lambda1 * w + self.lr / labels.size * gradient
            self.log_like(data, labels, w, z, prob)
            if step > 300:
                if -log_likelihood[step - 1] + log_likelihood[step] < self.stop_dif:
                    break
        return w
X_train = _input1.drop(['Transported'], axis=1)
X_train = X_train.to_numpy(dtype='float32')
Y_train = _input1['Transported'].to_numpy(dtype='int')
X_train = X_train - np.mean(X_train, axis=0)
X_train = X_train / np.std(X_train, axis=0)
model = LogisticRegression(X_train, Y_train, 0.1, 1000, 0.001)
fit = model.log_reg(X_train, Y_train)
pred_train = model.predict(X_train, fit)
accuracy_train_x = (pred_train == Y_train).mean()
predicted_labels_test = model.predict(_input0.to_numpy(dtype='float32'), fit)
plt.figure()
plt.title('Likelihood vs Iterations ')
plt.xlabel('Iterations')
plt.ylabel('Log-Likelihood')
plt.plot(range(len(log_likelihood)), log_likelihood, color='r')
plt.legend()
plt.grid()
accuracy_train_x