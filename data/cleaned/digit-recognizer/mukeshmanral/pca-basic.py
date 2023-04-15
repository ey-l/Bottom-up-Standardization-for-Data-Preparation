import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
d0 = pd.read_csv('data/input/digit-recognizer/train.csv')
d0.head()
l = d0.label
d = d0.drop('label', axis=1)
print(l.shape)
print(d.shape)
idx = 1
print(l[idx])
d.head()
plt.figure(figsize=(2, 2))
grid_data = d.loc[idx].values.reshape(28, 28)
plt.imshow(grid_data, interpolation='none', cmap='gray')

labels = l.head(15000)
data = d.head(15000)
print('the shape of sample data = ', data.shape)
d.head()
l.head()
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)
data.shape
sample_data = standardized_data
covar_matrix = np.matmul(sample_data.T, sample_data)
print('The shape of variance matrix \n', covar_matrix.shape)
covar_matrix
from scipy.linalg import eigh
(values, vectors) = eigh(covar_matrix, eigvals=(782, 783))
print('Shape of eigen vectors = ', vectors.shape)
print(vectors)
vectors = vectors.T
print('Updated shape of eigen vectors = ', vectors.shape)
print(vectors)
new_coordinates = np.matmul(vectors, sample_data.T)
print(" resultanat new data points' shape ", vectors.shape, 'X', sample_data.T.shape, ' = ', new_coordinates.shape)
new_coordinates = np.vstack((new_coordinates, labels)).T
dataframe = pd.DataFrame(data=new_coordinates, columns=('1st_principal', '2nd_principal', 'label'))
dataframe.head()
sns.FacetGrid(dataframe, hue='label', height=8).map(plt.scatter, '1st_principal', '2nd_principal', 'label').add_legend()

from sklearn import decomposition
pca = decomposition.PCA()
pca.n_components = 2
pca_data = pca.fit_transform(sample_data)
print('shape of pca_reduced.shape = ', pca_data.shape)
pd.DataFrame(data=np.vstack((pca_data.T, labels)).T, columns=('1st_principal', '2nd_principal', 'label'))
pca_data = np.vstack((pca_data.T, labels)).T
pca_df = pd.DataFrame(data=pca_data, columns=('1st_principal', '2nd_principal', 'label'))
sns.FacetGrid(pca_df, hue='label', height=8).map(plt.scatter, '1st_principal', '2nd_principal', 'label').add_legend()

pca.n_components = 784
pca_data = pca.fit_transform(sample_data)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
cumulative_explained_variance = np.cumsum(percentage_var_explained)
plt.figure(figsize=(6, 4))
plt.plot(cumulative_explained_variance, linewidth=4)
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
