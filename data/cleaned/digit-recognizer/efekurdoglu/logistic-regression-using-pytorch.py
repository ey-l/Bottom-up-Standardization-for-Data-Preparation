import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
train = pd.read_csv('data/input/digit-recognizer/train.csv', dtype=np.float32)
targets_numpy = train.label.values
features_numpy = train.loc[:, train.columns != 'label'].values / 255
(features_train, features_test, targets_train, targets_test) = train_test_split(features_numpy, targets_numpy, test_size=0.2, random_state=42)
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
plt.imshow(features_numpy[10].reshape(28, 28))
plt.axis('off')
plt.title(str(targets_numpy[10]))
plt.savefig('graph.png')


class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
input_dim = 28 * 28
output_dim = 10
model = LogisticRegressionModel(input_dim, output_dim)
error = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
count = 0
loss_list = []
iteration_list = []
for epoch in range(num_epochs):
    for (i, (images, labels)) in enumerate(train_loader):
        train = Variable(images.view(-1, 28 * 28))
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
            for (images, labels) in test_loader:
                test = Variable(images.view(-1, 28 * 28))
                outputs = model(test)
                predicted = torch.max(outputs.data, 1)[1]
                total += len(labels)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / float(total)
            loss_list.append(loss.data)
            iteration_list.append(count)
        if count % 500 == 0:
            print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))
plt.plot(iteration_list, loss_list)
plt.xlabel('Number of iteration')
plt.ylabel('Loss')
plt.title('Logistic Regression: Loss vs Number of iteration')
