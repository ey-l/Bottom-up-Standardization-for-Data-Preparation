import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
input_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.ff1 = nn.Linear(8, 42)
        self.ff2 = nn.Linear(42, 50)
        self.out = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.ff1(x))
        x = F.relu(self.ff2(x))
        x = self.out(x)
        return x
features = input_data.drop(['Outcome'], axis=1)
target = input_data['Outcome']
feature_tensor = torch.FloatTensor(features.values)
target_tensor = torch.LongTensor(target.values)
(x_train, x_test, y_train, y_test) = train_test_split(feature_tensor, target_tensor, test_size=0.2, random_state=1)
torch.manual_seed(29)
model = NeuralNetwork()
model.parameters
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)
epochs = 500
final_losses = []
for i in range(epochs):
    i += 1
    y_pred = model.forward(x_train)
    loss = loss_function(y_pred, y_train)
    final_losses.append(loss.item())
    if i % 100 == 1:
        print(f'At epoch {i}, the loss is {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs), final_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
y_pred = model(x_test)
predictions = []
with torch.no_grad():
    for data in x_test:
        y_pred = model(data).argmax().item()
        predictions.append(y_pred)
accuracy_score(predictions, y_test)
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True)
torch.save(model, 'pima.pt')
load_model = torch.load('pima.pt')