import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
d0 = pd.read_csv('data/input/digit-recognizer/train.csv')
print(d0.head(5))
l = d0['label']
d = d0.drop('label', axis=1)
print(d.shape)
print(l.shape)
plt.figure(figsize=(7, 7))
idx = 1
grid_data = d.iloc[idx].to_numpy().reshape(28, 28)
plt.imshow(grid_data, interpolation='none', cmap='gray')

print(l[idx])
labels = l.head(15000)
data = d.head(15000)
print('The shape of sample data = ', data.shape)
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)
print(standardized_data.shape)
import seaborn as sns
data_1000 = standardized_data[0:1000, :]
labels_1000 = labels[0:1000]
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))
sns.FacetGrid(tsne_df, hue='label', height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

model = TSNE(n_components=2, random_state=0, perplexity=50)
tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))
sns.FacetGrid(tsne_df, hue='label', height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With Perplexity 50')

model = TSNE(n_components=2, random_state=0, perplexity=50, n_iter=5000)
tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))
sns.FacetGrid(tsne_df, hue='label', height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With Perplexity 50')

model = TSNE(n_components=2, random_state=0, perplexity=2)
tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))
sns.FacetGrid(tsne_df, hue='label', height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With Perplexity 50')
