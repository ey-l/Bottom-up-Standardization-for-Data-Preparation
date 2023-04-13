import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
train_data.head()
train_data.info()

def divide_convert(age, num=10):
    return int(age / num)
train_data['Age'] = train_data['Age'].apply(lambda x: divide_convert(x))
train_data['SkinThickness'] = train_data['SkinThickness'].apply(lambda x: divide_convert(x))
train_data.head()
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x = train_data.iloc[:-20, :-1].to_numpy()
x = torch.FloatTensor(x).to(device)
y = train_data.iloc[:-20, -1].to_numpy()
y = torch.FloatTensor(y).to(device).reshape(-1, 1)
valid_x = train_data.iloc[-20:, :-1].to_numpy()
valid_x = torch.FloatTensor(valid_x).to(device)
valid_y = train_data.iloc[-20:, -1].to_numpy()
valid_y = torch.FloatTensor(valid_y).to(device).reshape(-1, 1)
print(x.shape, y.shape)
print(valid_x.shape, valid_y.shape)
import torch.nn as nn
import torch.nn.functional as F

class BinaryDnnModel(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(features, features * 2), nn.LeakyReLU(), nn.Linear(features * 2, features), nn.LeakyReLU(), nn.Linear(features, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)
model = BinaryDnnModel(x.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
n_epoch = 500
model.train()
for epoch in range(n_epoch):
    predict = model(x)
    loss = F.binary_cross_entropy(predict, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    predict = predict >= torch.FloatTensor([0.5]).to(device)
    predict = predict.float()
    acc = predict == y
    acc = acc.sum()
    acc = acc / x.shape[0]
    print('Epoch : {}/{},   loss : {:.5f},    acc : {:.5f}'.format(epoch + 1, n_epoch, loss.item(), acc))
model.eval()
valid_acc = 0
valid_loss = 0
for i in range(20):
    predict = model(valid_x[i])
    loss = F.binary_cross_entropy(predict, valid_y[i])
    valid_loss += loss.item() / 20
    predict = predict >= torch.FloatTensor([0.5]).to(device)
    predict = predict.float()
    if predict.item() == valid_y[i].item():
        valid_acc += 1
print('acc : {0}, loss : {1:.5f}'.format(valid_acc / 20, valid_loss))