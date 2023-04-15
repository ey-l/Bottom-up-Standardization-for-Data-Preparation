import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
import torch
import torch.nn.functional as f
import torch
from torch.autograd import Variable
import itertools
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
train.head(3)
y_train = train[['label']]
x_train = train.loc[:, train.columns != 'label']
x_train.head(3)
y_train.head(3)
ax = sns.countplot(x='label', data=y_train, palette='Set3')
x_t = x_train.to_numpy().reshape((-1, 28, 28))
y_t = y_train.to_numpy().reshape(-1)
x_t.shape
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_t[i], cmap='gray')
    plt.title('value of frame:' + str(y_t[i]))
    plt.xticks([])
Y_train = train.label.values
X_train = x_train / 255
X_test = test / 255
i_test = [i + 1 for i in range(len(X_test))]
X_test
X_train = np.array(X_train)
X_test = np.array(X_test)
(X_train, X_val, Y_train, Y_val) = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
X_train_s = X_train
Y_train_s = Y_train
X_train = torch.Tensor(X_train)
X_val = torch.Tensor(X_val)
X_test = torch.Tensor(X_test)
i_test = torch.Tensor(i_test)
Y_train = torch.Tensor(Y_train).type(torch.LongTensor)
Y_val = torch.Tensor(Y_val).type(torch.LongTensor)
X_test.shape

class CNN_Model(nn.Module):

    def __init__(self):
        super(CNN_Model, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
batch_size = 100
n_iters = 2500
num_epochs = n_iters / (len(X_train) / batch_size)
num_epochs = int(num_epochs)
trn = torch.utils.data.TensorDataset(X_train, Y_train)
val = torch.utils.data.TensorDataset(X_val, Y_val)
tst = torch.utils.data.TensorDataset(X_test, i_test)
train_loader = torch.utils.data.DataLoader(trn, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(tst, batch_size=batch_size, shuffle=False)
model = CNN_Model()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
error = nn.CrossEntropyLoss()
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for (i, (images, labels)) in enumerate(train_loader):
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(train)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()
        count += 1
        if count % 50 == 0:
            correct = 0
            total = 0
            for (images, labels) in val_loader:
                val = Variable(images.view(100, 1, 28, 28))
                outputs = model(val)
                predicted = torch.max(outputs.data, 1)[1]
                total += len(labels)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / float(total)
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
type(test_loader)
f = plt.figure(figsize=(20, 6))
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)
ax.plot(iteration_list, loss_list, color='red')
ax.set_xlabel('Number of iteration')
ax.set_ylabel('Loss')
ax.set_title('CNN: Loss vs Number of iteration')
ax2.plot(iteration_list, accuracy_list, color='green')
ax2.set_xlabel('Number of iteration')
ax2.set_ylabel('Accuracy')
ax2.set_title('CNN: Accuracy vs Number of iteration')
test_outputs = []
for (images, index) in test_loader:
    val = Variable(torch.Tensor(np.array(images).reshape(100, 1, 28, 28)))
    outputs = model(val)
    predicted = torch.max(outputs.data, 1)[1]
    predicted = predicted.tolist()
    test_outputs.append(predicted)
test_outputs = list(itertools.chain.from_iterable(test_outputs))
len(test_outputs)
Label = pd.Series(test_outputs, name='Label').astype(int)
ImageId = pd.Series(i_test, name='ImageId').astype(int)
results = pd.concat([ImageId, Label], axis=1)

results.head()
classifier = svm.SVC(gamma=0.001)