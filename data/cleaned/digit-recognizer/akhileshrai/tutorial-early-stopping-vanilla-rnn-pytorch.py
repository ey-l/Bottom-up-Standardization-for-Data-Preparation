import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
train_dataset = pd.read_csv('data/input/digit-recognizer/train.csv', dtype=np.float32)
test_dataset = pd.read_csv('data/input/digit-recognizer/test.csv', dtype=np.float32)
targets_numpy = train_dataset.label.values
features_numpy = train_dataset.loc[:, train_dataset.columns != 'label'].values / 255
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
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class RNNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        (out, hn) = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out
batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
input_dim = 28
hidden_dim = 100
layer_dim = 3
output_dim = 10
model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
error = nn.CrossEntropyLoss()
learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
seq_dim = 28
loss_list = []
iteration_list = []
accuracy_list = []
count = 0
min_val_loss = np.Inf
val_array = []
correct = 0
iter = 0
count = 0
iter_array = []
loss_array = []
total = 0
accuracy_array = []
n_epochs_stop = 6
epochs_no_improve = 0
early_stop = False
for epoch in range(num_epochs):
    val_loss = 0
    for (i, (images, labels)) in enumerate(train_loader):
        train = Variable(images.view(-1, seq_dim, input_dim))
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(train)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()
        val_loss += loss
        val_loss = val_loss / len(train_loader)
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
        else:
            epochs_no_improve += 1
        iter += 1
        if epoch > 5 and epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            early_stop = True
            break
        else:
            continue
        break
        if iter % 336 == 0:
            correct = 0
            total = 0
    if early_stop:
        print('Stopped')
        break
    for (images, labels) in test_loader:
        images = images.view(-1, seq_dim, input_dim)
        outputs = model(images)
        (_, predicted) = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        accuracy = 100 * correct / total
        count = count + 1
        if iter % 336 == 0 and count % 100 == 0:
            iter_array.append(iter)
            loss_array.append(loss.item())
            accuracy_array.append(accuracy.item())
            print('Epoch: {}. Iteration: {}. Loss: {}. Accuracy: {}, Count: {}'.format(epoch, iter, loss.item(), accuracy.item(), count))
examples = enumerate(test_loader)
(batch_idx, (images, labels)) = next(examples)
images = images.numpy()
labels = labels.numpy()
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(images[i].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title('Number: {}'.format(labels[i]))
    plt.xticks([])
    plt.yticks([])
print(fig)
df = pd.DataFrame({'Iterations': iter_array, 'Loss': loss_array, 'Accuracy': accuracy_array})
df['Index'] = range(1, len(iter_array) + 1)
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.models import CustomJS, ColumnDataSource, Select, HoverTool, LinearInterpolator, Column
from bokeh.layouts import column
from bokeh.models.widgets import Div
output_notebook()
source_CDS = ColumnDataSource(df)
df
hover = HoverTool(tooltips='@Loss= Loss')
Loss_line = figure(plot_width=700, plot_height=300, tools=[hover])
Loss_line.line('Iterations', 'Loss', source=source_CDS, line_width=2)
Loss_line.background_fill_color = '#fffce6'
title_div = Div(text='<b> Loss vs Iterations </b>', style={'font-size': '400%', 'color': '#FF6347'})
p2 = column(title_div, Loss_line)
show(p2)
hover = HoverTool(tooltips=' Accuracy: @Accuracy%')
Accuracy_line = figure(plot_width=700, plot_height=300, tools=[hover])
Accuracy_line.line('Iterations', 'Accuracy', source=source_CDS, line_width=2)
title_div2 = Div(text='<b> Accuracy vs Iterations </b>', style={'font-size': '400%', 'color': '#008080'})
Accuracy_line.background_fill_color = '#fffce6'
p2 = column(title_div2, Accuracy_line)
show(p2)
test_dataset = pd.read_csv('data/input/digit-recognizer/test.csv', dtype=np.float32)
test_dataset.shape
test_dataset = torch.from_numpy(test_dataset.values)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False)
model.eval()
test_pred = torch.LongTensor()
for (i, data) in enumerate(test_loader):
    predict = data.view(-1, seq_dim, input_dim)
    predict = Variable(predict)
    output = model(predict)
    pred = output.data.max(1, keepdim=True)[1]
    test_pred = torch.cat((test_pred, pred), dim=0)
test_pred.size()
Submission_df = pd.DataFrame(np.c_[np.arange(1, len(test_pred.numpy()) + 1)[:, None], test_pred.numpy()], columns=['ImageId', 'Label'])
print(Submission_df.head())

test_dataset2 = pd.read_csv('data/input/digit-recognizer/test.csv', dtype=np.float32)
test_dataset2 = test_dataset2.values
plt.imshow(test_dataset2[4].reshape(28, 28))
print(test_pred[4])
test_pred
from collections import Counter
list = Counter(Submission_df['Label'].values)
list
train_dataset2 = pd.read_csv('data/input/digit-recognizer/train.csv', dtype=np.float32)
train_dataset2 = train_dataset2['label'].values
list = Counter(train_dataset2)
list