import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
pass
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(X.shape, '\n', y.shape)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape, '\n', X_test.shape, y_test.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

class ANN_Model(nn.Module):

    def __init__(self, input_features=8, hidden1=20, hidden2=20, out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_features)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x
torch.manual_seed(20)
model = ANN_Model()
model.parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 500
final_losses = []
for i in range(epochs):
    i = i + 1
    y_pred = model.forward(X_train)
    loss = loss_fn(y_pred, y_train)
    final_losses.append(loss)
    if i % 10 == 1:
        print(f'Epoch number {i} and loss: {loss.item()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
fi_loss = [fl.item() for fl in final_losses]
pass
pass
pass
predictions = []
with torch.no_grad():
    for (i, data) in enumerate(X_test):
        y_pred = model(data)
        predictions.append(y_pred.argmax().item())
        print(y_pred.argmax().item())
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
cm
pass
pass
pass
pass
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, predictions)
score
torch.save(model, 'diabetes.pt')
model = torch.load('./diabetes.pt')
model.eval()
new = list(df.iloc[0, :-1])
new
new_data = torch.tensor(new)
new_data
with torch.no_grad():
    print(model(new_data).argmax().item())