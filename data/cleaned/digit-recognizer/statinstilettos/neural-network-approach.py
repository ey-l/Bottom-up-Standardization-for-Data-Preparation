import numpy as np
import pandas as pd
np.random.seed(1)
test = pd.read_csv('data/input/digit-recognizer/test.csv')
train = pd.read_csv('data/input/digit-recognizer/train.csv')
print(train.shape)
print(test.shape)
train.head()
import matplotlib.pyplot as plt
plt.hist(train['label'])
plt.title('Frequency Histogram of Numbers in Training Data')
plt.xlabel('Number Value')
plt.ylabel('Frequency')
import math
(f, ax) = plt.subplots(5, 5)
for i in range(1, 26):
    data = train.iloc[i, 1:785].values
    (nrows, ncols) = (28, 28)
    grid = data.reshape((nrows, ncols))
    n = math.ceil(i / 5) - 1
    m = [0, 1, 2, 3, 4] * 5
    ax[m[i - 1], n].imshow(grid)
label_train = train['label']
train = train.drop('label', axis=1)
train = train / 255
test = test / 255
train['label'] = label_train
from sklearn import decomposition
from sklearn import datasets
pca = decomposition.PCA(n_components=200)