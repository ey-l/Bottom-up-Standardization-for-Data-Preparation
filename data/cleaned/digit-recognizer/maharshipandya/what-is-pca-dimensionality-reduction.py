import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/digit-recognizer/train.csv')
df
train = df.drop(columns=['label'])
labels = df['label']
print('Training set shape: ', train.shape)
print('Target variable shape:', labels.shape)

def plot_digits(data, images_per_row=5):
    for i in range(len(data)):
        ax = plt.subplot(images_per_row, images_per_row, i + 1)
        digit = data[i]
        digit_img = digit.reshape(28, 28)
        plt.imshow(digit_img, cmap=plt.cm.Blues)
        plt.axis('off')
plot_digits(train[:25].to_numpy(), images_per_row=5)

pca = PCA(n_components=0.95)
train_reduced = pca.fit_transform(train)
print('New shape of the training set: ', train_reduced.shape)
print('Dimensionality reduced to: ', pca.components_.T.shape[1], 'dimensions')
s = 0
for (i, evr) in enumerate(pca.explained_variance_ratio_):
    print(f'Ratio of variance preserved by PC{i + 1}: ', evr)
    s += evr
print('\nTotal variance preserved: ', s)
train_recovered = pca.inverse_transform(train_reduced)
plot_digits(train_recovered[:25], images_per_row=5)
