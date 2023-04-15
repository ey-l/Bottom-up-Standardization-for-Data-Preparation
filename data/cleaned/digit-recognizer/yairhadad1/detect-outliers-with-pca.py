import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
sns.set()
train = pd.read_csv('data/input/digit-recognizer/train.csv')
X = train.drop('label', axis=1)
y = train['label'].astype('category')
X.head()
dataset_7 = X[y == 7].reset_index().drop('index', axis=1)
plt.figure(figsize=(10, 8))
(row, colums) = (3, 3)
for i in range(9):
    plt.subplot(colums, row, i + 1)
    plt.imshow(dataset_7.iloc[i].values.reshape(28, 28), interpolation='nearest', cmap='Greys')

n_components = 5
pca = PCA(n_components=n_components)
pca_dataset_7 = pca.fit_transform(dataset_7)
inverse_transform_dataset_7 = pca.inverse_transform(pca_dataset_7)
print('dataset_7 shape', dataset_7.shape)
print('pca_dataset_7 shape', pca_dataset_7.shape)
print('inverse_transform_dataset_7 shape', inverse_transform_dataset_7.shape)
MSE_score = ((dataset_7 - inverse_transform_dataset_7) ** 2).sum(axis=1)
MSE_score.head()
MSE_max_scores = MSE_score.nlargest(9).index
plt.figure(figsize=(10, 8))
(row, colums) = (3, 3)
for i in range(9):
    plt.subplot(colums, row, i + 1)
    plt.imshow(dataset_7.iloc[MSE_max_scores[i]].values.reshape(28, 28), interpolation='nearest', cmap='Greys')

plt.figure(figsize=(10, 8))
(row, colums) = (5, 10)
for number in range(10):
    dataset = pd.DataFrame(X[y == number].reset_index().drop('index', axis=1))
    pca = PCA(n_components=n_components)
    pca_dataset = pca.fit_transform(dataset)
    inverse_transform_dataset = pca.inverse_transform(pca_dataset)
    MSE_score = ((dataset - inverse_transform_dataset) ** 2).sum(axis=1)
    MSE_worst = MSE_score.nlargest(5).index
    for number2 in range(0, 5):
        plt.subplot(colums, row, number2 + number * 5 + 1)
        plt.imshow(dataset.iloc[MSE_worst[number2]].values.reshape(28, 28), interpolation='nearest', cmap='Greys')
