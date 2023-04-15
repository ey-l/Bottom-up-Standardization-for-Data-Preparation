import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib

from sklearn.decomposition import PCA
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/digit-recognizer/train.csv')
df.head()
df.shape
test_df = df['label']
train_df = df.drop('label', axis=1)
from sklearn.preprocessing import StandardScaler
X = train_df.values
X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
(eig_vals, eig_vecs) = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
tot = sum(eig_vals)
var_exp = [i / tot * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print(var_exp)
print(cum_var_exp)
plt.plot(list(range(784)), cum_var_exp, label='Cumulative Variances')
plt.xlabel('Feature No.')
plt.ylabel('Variance %')
plt.title('Cumulative Variance % v/s Features')

plt.plot(list(range(784)), var_exp, label='Individual Variances')
plt.xlabel('Feature No.')
plt.ylabel('Variance %')
plt.title('Individual Variance % v/s Features')

n_components = 300