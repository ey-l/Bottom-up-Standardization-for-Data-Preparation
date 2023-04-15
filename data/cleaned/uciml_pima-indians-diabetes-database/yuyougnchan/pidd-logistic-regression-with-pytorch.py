import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y, 'g')
plt.plot([0, 0], [1.0, 0.0], ':')
plt.title('Sigmoid Function')

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5 * x)
y2 = sigmoid(x)
y3 = sigmoid(2 * x)
plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b', linestyle='--')
plt.plot([0, 0], [1.0, 0.0], ':')
plt.title('Sigmoid Function')

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x + 0.5)
y2 = sigmoid(x + 1)
y3 = sigmoid(x + 1.5)
plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b', linestyle='--')
plt.plot([0, 0], [1.0, 0.0], ':')
plt.title('Sigmoid Function')

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.autograd import Variable
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
(X_train, y_train) = (df.drop(['Outcome'], axis=1), df['Outcome'])
print(X_train.shape)
print(y_train.shape)

class LogisticRegression(nn.Module):

    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.f1 = nn.Linear(input_dim, 1000)
        self.f2 = nn.Linear(1000, output_dim)

    def forward(self, x):
        x = self.f1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.2)
        x = self.f2(x)
        return F.sigmoid(x)
batch_size = 16
batch_no = len(X_train) // batch_size

def generate_batches(X, y, batch_size):
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))
    for i in range(len(X) // batch_size):
        if i + batch_size >= len(X):
            continue
        ind = perm[i * batch_size:(i + 1) * batch_size]
        yield (X[ind], y[ind])
input_dim = 8
output_dim = 2
learning_rate = 1e-05
model = LogisticRegression(input_dim, output_dim)
error = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_list = []
roc_list = []
iteration_number = 200
for iteration in range(iteration_number):
    batch_loss = 0
    batch_roc = 0
    size = 0
    for (x, y) in generate_batches(X_train, y_train, batch_size):
        inputs = Variable(torch.from_numpy(x)).float()
        labels = Variable(torch.from_numpy(y))
        optimizer.zero_grad()
        results = model(inputs)
        loss = error(results, labels)
        batch_loss += loss.data
        loss.backward()
        optimizer.step()
        batch_roc += roc_auc_score(labels.detach().numpy(), results[:, 1].detach().numpy())
        size += 1
    loss_list.append(batch_loss / batch_no)
    roc_list.append(batch_roc / size)
    if iteration % 10 == 0:
        print('Epoch {}: loss {}, ROC {}'.format(iteration, batch_loss / batch_no, batch_roc / size))
plt.plot(range(iteration_number), loss_list)
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')

plt.plot(range(iteration_number), roc_list)
plt.xlabel('Number of Iterations')
plt.ylabel('ROC')
