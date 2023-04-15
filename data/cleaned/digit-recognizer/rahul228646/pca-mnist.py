import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from sklearn import decomposition
train_df = pd.read_csv('data/input/digit-recognizer/train.csv')
train_df.head()
label = train_df.label
train = train_df.drop('label', axis=1)
print(label.shape)
print(train.shape)
label[0]
train.head()
plt.figure(figsize=(2, 2))
grid_data = train.loc[0].values.reshape(28, 28)
plt.imshow(grid_data, interpolation='none', cmap='gray')

standardized_data = StandardScaler().fit_transform(train)
standardized_data.shape
covar_matrix = np.matmul(standardized_data.T, standardized_data)
covar_matrix
(values, vectors) = eigh(covar_matrix, eigvals=(782, 783))
print('Shape of eigen vectors = ', vectors.shape)
print(vectors)
vectors = vectors.T
print('Updated shape of eigen vectors = ', vectors.shape)
print(vectors)
new_coord = np.matmul(vectors, standardized_data.T)
print(new_coord)
print(new_coord.shape)
pca_data = pd.DataFrame({'1st_principal': new_coord[1], '2nd_principal': new_coord[0], 'label': label})
pca_data
sns.FacetGrid(pca_data, hue='label', height=8).map(plt.scatter, '1st_principal', '2nd_principal', 'label').add_legend()

pca = decomposition.PCA()
pca.n_components = 2
pca_data_sci = pca.fit_transform(standardized_data)
pca_data_sci.shape
pca_data_sci_new = pd.DataFrame({'1st_principal': pca_data_sci.T[0], '2nd_principal': pca_data_sci.T[1], 'label': label})
pca_data_sci_new
sns.FacetGrid(pca_data_sci_new, hue='label', height=8).map(plt.scatter, '1st_principal', '2nd_principal', 'label').add_legend()

pca.n_components = 784
pca_data = pca.fit_transform(standardized_data)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
cumulative_explained_variance = np.cumsum(percentage_var_explained)
plt.figure(figsize=(6, 4))
plt.plot(cumulative_explained_variance, linewidth=3)
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
