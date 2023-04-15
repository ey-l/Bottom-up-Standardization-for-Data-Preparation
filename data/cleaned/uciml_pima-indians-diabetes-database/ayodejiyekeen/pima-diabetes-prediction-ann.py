import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isna().sum()
diabetes = df.copy()
diabetes['Outcome'] = np.where(df['Outcome'] == 1, 'Diabetic', 'Non-Daibetic')
diabetes.head()
sns.pairplot(data=diabetes, hue='Outcome')
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
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
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
EPOCH = 500
final_losses = []
for i in range(EPOCH):
    i += 1
    y_pred = model.forward(X_train)
    loss = loss_func(y_pred, y_train)
    final_losses.append(loss.item())
    if i % 50 == 1:
        print(f'Epoch number: {i} and the loss : {loss.item()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(EPOCH), final_losses)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')

predictions = []
with torch.no_grad():
    for (i, data) in enumerate(X_test):
        y_pred = model(data)
        predictions.append(y_pred.argmax().item())
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, predictions)
cm
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

score = accuracy_score(y_test, predictions)
score
torch.save(model, 'diabetes.pt')