import numpy as np
import pandas as pd
from fastai.vision.all import *
from tqdm import tqdm
import pickle, gzip, math, torch, matplotlib as mpl
import torch.nn.functional as F
import pathlib
from pathlib import Path
from fastai.callback.fp16 import *
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
training_df_all = pd.read_csv('data/input/digit-recognizer/train.csv')
testing_df_all = pd.read_csv('data/input/digit-recognizer/test.csv')
sample_submission = pd.read_csv('data/input/digit-recognizer/sample_submission.csv')
valid_df = training_df_all[35000:42000]
training_df = training_df_all[0:34999]
training_df.tail()
x_train = torch.tensor(training_df.drop(['label'], axis=1).values).float() / 255.0
y_train = torch.tensor(training_df.label.values).float().unsqueeze(1)
x_valid = torch.tensor(valid_df.drop(['label'], axis=1).values).float() / 255.0
y_valid = torch.tensor(valid_df.label.values).float().unsqueeze(1)
print(x_train.shape, y_train.shape)
x_train_image = torch.reshape(x_train, (-1, 28, 28))
print(x_train_image.shape)
show_image(x_train_image[0])
plt.imshow(x_train_image[0])
dset = list(zip(x_train, y_train.squeeze(1).long()))
valid_dset = list(zip(x_valid, y_valid.squeeze(1).long()))
(x, y) = dset[0]
(x.shape, y)
dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)
(xb, yb) = first(dl)
(xb.shape, yb.shape)

def init_params(size, std=1.0):
    return (torch.randn(size) * std).requires_grad_()
weights = init_params((28 * 28, 10))
bias = init_params(10)
(weights[0], bias[0])

def simple_net(xb):
    res = xb @ w1 + b1
    res = res.max(tensor(0.0))
    res = res @ w2 + b2
    return res
w1 = init_params((28 * 28, 50))
b1 = init_params(50)
w2 = init_params((50, 10))
b2 = init_params(10)
preds = simple_net(x_train)
(preds[0], preds.shape)
loss_func = F.cross_entropy

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = loss_func(preds, yb)
    loss.backward()

def train_epoch(model, lr, params):
    for (xb, yb) in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()

def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()

def validate_epoch(model):
    accs = [accuracy(model(xb), yb) for (xb, yb) in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)
preds = simple_net(xb)
print(preds[0], preds.shape)
print(loss_func(preds, yb))
print(accuracy(preds, yb))
lr = 1
params = (w1, b1, w2, b2)
train_epoch(simple_net, lr, params)
validate_epoch(simple_net)
for i in range(20):
    train_epoch(simple_net, lr, params)
    print(validate_epoch(simple_net), end=' ')
simple_net = nn.Sequential(nn.Linear(28 * 28, 50), nn.ReLU(), nn.Linear(50, 10))
(w1, b1, w2, b2) = simple_net.parameters()
print(w1.shape, b1.shape, w2.shape, b2.shape)
opt = SGD(simple_net.parameters(), lr)

def train_epoch(model):
    for (xb, yb) in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()
validate_epoch(simple_net)

def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')
train_model(simple_net, 20)
dls = DataLoaders(dl, valid_dl)
learn = Learner(dls, simple_net, opt_func=SGD, loss_func=F.cross_entropy, metrics=accuracy)