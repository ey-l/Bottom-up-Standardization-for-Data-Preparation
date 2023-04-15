import pandas as pd
train_data = pd.read_csv('data/input/digit-recognizer/train.csv')
test_data = pd.read_csv('data/input/digit-recognizer/test.csv')
train_data.head()
import matplotlib.pyplot as plt
plt.figure(figsize=(30, 30))
for i in range(10):
    plt.subplot(20, 20, i + 1)
    plt.title('No.' + str(i))
    plt.imshow(train_data.iloc[:, 1:].iloc[i].values.reshape(28, 28), cmap='Greys')
from sklearn.model_selection import train_test_split
import numpy as np
x = np.array(train_data.iloc[:, 1:785])
y = np.array(train_data['label'])
(x_train, x_valid, y_train, y_valid) = train_test_split(x, y, test_size=0.3, random_state=42)
from sklearn.neural_network import MLPClassifier
deep_learning = MLPClassifier(random_state=123, verbose=True)