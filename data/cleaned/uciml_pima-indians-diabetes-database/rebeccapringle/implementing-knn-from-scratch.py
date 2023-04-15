import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.head()
X = df.drop('target', axis=1)
y = df.target

def distance(point1, point2):
    squared_difference = 0
    for i in range(len(point1)):
        squared_difference += (point1[i] - point2[i]) ** 2
    final_distance = squared_difference ** 0.5
    return final_distance
distance(X.iloc[0], X.iloc[1])
test_pt = [4.8, 2.7, 2.5, 0.7]
distances = []
for i in X.index:
    distances.append(distance(test_pt, X.iloc[i]))
df_dists = pd.DataFrame(data=distances, index=X.index, columns=['distance'])
df_nn = df_dists.sort_values(by=['distance'], axis=0)[:5]
df_nn
from collections import Counter
counter = Counter(y[df_nn.index])
counter.most_common()[0][0]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def classify(test_point, X_train, y_train, k):
    distances = []
    for point in X_train:
        distance_to_point = distance(point, test_point)
        distances.append(distance_to_point)
    df_dists = pd.DataFrame(data=distances, columns=['dist'], index=y_train.index)
    df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]
    counter = Counter(y_train[df_nn.index])
    prediction = counter.most_common()[0][0]
    return prediction
classify([5.2, 3.4, 2.4, 0.9], X_train, y_train, k=5)

def knn_predict(X_test, X_train, y_train, k):
    y = []
    for test in X_test:
        prediction = classify(test, X_train, y_train, k)
        y.append(prediction)
    return y
knn_predict(X_test, X_train, y_train, k=5)
from sklearn.metrics import accuracy_score
accuracies = []
for k in range(1, 100):
    y_hat_test = knn_predict(X_test, X_train, y_train, k)
    accuracies.append(accuracy_score(y_test, y_hat_test))
(fig, ax) = plt.subplots(figsize=(8, 6))
ax.plot(range(1, 100), accuracies)
ax.set_xlabel('# of Nearest Neighbors (k)')
ax.set_ylabel('Accuracy (%)')