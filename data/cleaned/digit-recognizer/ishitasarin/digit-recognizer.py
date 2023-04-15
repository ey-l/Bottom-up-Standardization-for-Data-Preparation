import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
sample_submission = pd.read_csv('data/input/digit-recognizer/sample_submission.csv')
train.head()
test.head()
train.shape
x = train.drop('label', axis=1).to_numpy()
y = train['label'].to_numpy()
x = x / 255
randomdig = x[404]
randomdig_img = randomdig.reshape((28, 28))
plt.figure()
plt.imshow(randomdig_img, cmap='gray')

from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)
x_train
from sklearn.neighbors import KNeighborsClassifier as knn

def elbow(k):
    error_test = []
    for i in k:
        model = knn(n_neighbors=i)