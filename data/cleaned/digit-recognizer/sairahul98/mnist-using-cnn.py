import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/digit-recognizer/train.csv')
test_df = pd.read_csv('data/input/digit-recognizer/test.csv')
(train_df.shape, test_df.shape)
train_df.head()
train_df.iloc[1, 1:].values.reshape(28, 28)
fig = plt.figure(figsize=(10, 10))
for i in range(1, 10):
    x = np.random.randint(1000)
    fig.add_subplot(3, 3, i)
    plt.title('Label: {}'.format(train_df.iloc[x, 0]))
    plt.imshow(train_df.iloc[x, 1:].values.reshape(28, 28), cmap='gray')
    plt.colorbar()
plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, x='label')
for c in range(0, 10):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Sample Class: {}'.format(c))
    data = train_df[train_df.label == c]
    for i in range(1, 10):
        fig.add_subplot(3, 3, i)
        plt.title('Label: {}'.format(data.iloc[i, 0]))
        plt.axis('off')
        plt.imshow(data.iloc[i, 1:].values.reshape(28, 28), cmap='cool')
X = train_df.iloc[:, 1:].values
Y = train_df.iloc[:, 0]
from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()