import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
plt.rcParams['image.cmap'] = 'gray'
color = ['#6388b4', '#ffae34', '#ef6f6a', '#8cc2ca', '#55ad89', '#c3bc3f', '#bb7693', '#baa094', '#a9b5ae', '#767676']
mnist = pd.read_csv('data/input/digit-recognizer/train.csv')
mnist.head()
label = mnist['label']
mnist.drop(['label'], inplace=True, axis=1)

def arr2img(arr, img_size=(28, 28)):
    return arr.reshape(img_size)
(fig, axes) = plt.subplots(2, 5, figsize=(10, 2))
for (idx, ax) in enumerate(axes.flat):
    ax.imshow(arr2img(mnist[idx:idx + 1].values))
    ax.set_title(label[idx], fontweight='bold', fontsize=8)
    ax.axis('off')
plt.subplots_adjust(bottom=0.1, right=0.5, top=0.9)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)