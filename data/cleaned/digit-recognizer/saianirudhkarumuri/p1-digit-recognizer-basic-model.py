import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)
from sklearn.model_selection import train_test_split
sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
train.info()
test.info()
y = train['label']
X = train.drop(labels=['label'], axis=1)
x_train_vis = np.array(X).reshape(X.shape[0], 28, 28)
(fig, axis) = plt.subplots(1, 4, figsize=(20, 10))
for (i, ax) in enumerate(axis.flat):
    ax.imshow(x_train_vis[i], cmap='binary')
    digit = y[i]
    ax.set(title=f'Real Number is {digit}')
y.value_counts()
sns.countplot(y)
(x_train, x_val, y_train, y_val) = train_test_split(X, y, test_size=0.1, random_state=2)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)