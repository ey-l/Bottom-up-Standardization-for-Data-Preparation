import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.read_csv('data/input/digit-recognizer/train.csv')
test_data = pd.read_csv('data/input/digit-recognizer/test.csv')
train_data.head()
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
X_train = X_train.reshape((-1, 28, 28))
X_test = X_test.reshape((-1, 28, 28))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i])

X_train = X_train.reshape((-1, 28 * 28))
X_test = X_test.reshape((-1, 28 * 28))
X_train = X_train / 256
X_test = X_test / 256
model = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64), early_stopping=True, verbose=True)