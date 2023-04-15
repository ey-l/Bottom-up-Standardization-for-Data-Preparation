import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from random import seed
seed(1)
kaggle = 1
if kaggle == 1:
    MNIST_PATH = 'data/input/digit-recognizer'
else:
    MNIST_PATH = '../Another_MNIST_try/data/input/digit-recognizer'

import os
for (dirname, _, filenames) in os.walk(MNIST_PATH):
    for filename in filenames:
        print(os.path.join(dirname, filename))
CSV_FILE_TRAIN = 'train.csv'
CSV_FILE_TEST = 'test.csv'

def load_mnist_data(minist_path, csv_file):
    csv_path = os.path.join(minist_path, csv_file)
    return pd.read_csv(csv_path)

def load_mnist_data_manuel(minist_path, csv_file):
    csv_path = os.path.join(minist_path, csv_file)
    csv_file = open(csv_path, 'r')
    csv_data = csv_file.readlines()
    csv_file.close()
    return csv_data

def split_train_val(data, val_ratio):
    return
train = load_mnist_data(MNIST_PATH, CSV_FILE_TRAIN)
test = load_mnist_data(MNIST_PATH, CSV_FILE_TEST)
train_2 = load_mnist_data_manuel(MNIST_PATH, CSV_FILE_TRAIN)
train.describe()
train.info()
train
train_copy = train.copy()
mnist_features = train_copy.drop('label', axis=1)
mnist_labels = train_copy['label']
plt.imshow(np.asfarray(mnist_features[4:5]).reshape(28, 28), cmap='binary')
plt.axis('off')


def print_digits(digit_dataframe):
    figsize = (8, 6)
    cols = 4
    rows = 6 // cols + 1

    def trim_axs(axs, N):
        """
        Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
        """
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]
    axs = plt.figure(figsize=figsize).subplots(rows, cols)
    axs = trim_axs(axs, len(digit_dataframe))
    i = 0
    for ax in axs:
        ax.imshow(np.asfarray(digit_dataframe.iloc[i]).reshape(28, 28), cmap='binary')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        i = i + 1
print_digits(mnist_features)
corr_matrix = train_copy.corr()
corr_matrix['label'].sort_values(ascending=False)
attributes = ['pixel381', 'pixel409', 'pixel436', 'pixel408']
pd.plotting.scatter_matrix(train_copy[attributes], figsize=(8, 6))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
num_pipeline = Pipeline([('std_scaler', StandardScaler())])
mnist_features_prepared = num_pipeline.fit_transform(mnist_features)
mnist_features_prepared
from sklearn.neighbors import KNeighborsClassifier
kneighbors = KNeighborsClassifier()