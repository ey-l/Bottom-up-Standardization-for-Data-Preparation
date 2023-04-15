import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
mnist = pd.read_csv('data/input/digit-recognizer/train.csv')
mnist
mnist.columns
mnist.head()
X = mnist.iloc[:, 1:].values
y = mnist['label'].values
X.shape
y
y.shape
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=12)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

def plot_image_color(img):
    img = img.reshape((28, 28))
    plt.imshow(img)

plot_image_color(X_train[60])

def plot_image(img):
    plt.imshow(img.reshape(28, 28), cmap='gray')

plot_image(X_train[60])
print(y_train[60])

def distance(pA, pB):
    return np.sum((pB - pA) ** 2) ** 0.5

def kNN(X, y, x_query, k=5):
    """
    X - > (m, 784)  np array (m is just number of eg. images)
    y - > (m,) np array
    x_query -> (1,874) np array
    k -> scaler  int
    
    do knn for classification
    """
    m = X.shape[0]
    distances = []
    for i in range(m):
        dis = distance(x_query, X[i])
        distances.append((dis, y[i]))
    distances = sorted(distances)
    distances = distances[:k]
    distances = np.array(distances)
    labels = distances[:, 1]
    (uniq_label, counts) = np.unique(labels, return_counts=True)
    pred = uniq_label[counts.argmax()]
    return int(pred)
kNN(X_train, y_train, X_test[235], k=7)
plot_image(X_test[235])
y_test[235]
predictions = []
for i in range(100):
    p = kNN(X_train, y_train, X_test[i], k=7)
    predictions.append(p)
predictions = np.array(predictions)
predictions.dtype
p = y_test[:100] == predictions
type(p)
(y_test[:100] == predictions).sum() / len(predictions)
predictions
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)