import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df['Outcome'] = np.where(df['Outcome'] == 1, 'Diabetic', 'Non Diabetic')
df
sns.pairplot(df, hue='Outcome')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
from sklearn.model_selection import train_test_split
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
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
        self.f_output = nn.Linear(hidden2, out_features)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.f_output(x)
        return x
torch.manual_seed(20)
model = ANN_Model()
model.parameters
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 500
final_losses = []
for i in range(epochs):
    i = i + 1
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred, y_train)
    final_losses.append(loss)
    if i % 10 == 1:
        print('For epoch {} loss is {} '.format(i, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    fi_los = [fl.item() for fl in final_losses]
import matplotlib.pyplot as plt

plt.plot(range(epochs), fi_los)

predictions = []
with torch.no_grad():
    for (i, data) in enumerate(X_test):
        y_pred = model(data)
        predictions.append(y_pred.argmax().item())
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
accuracy