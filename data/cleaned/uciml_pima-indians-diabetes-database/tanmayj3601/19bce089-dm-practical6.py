import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
df = data[['SkinThickness', 'BMI']]
df.head()
for i in df.columns:
    mean_ = np.mean(df[i])
    df[i] = df[i] - mean_
    print(mean_)
df.head()
cov_matrix = df.cov()
print(cov_matrix)
(eigen_value, eigen_vector) = np.linalg.eig(cov_matrix)
print(eigen_value)
print(eigen_vector)
ind = np.arange(0, len(eigen_value), 1)
ind = [x for (_, x) in sorted(zip(eigen_value, ind))][::-1]
eigen_value = eigen_value[ind]
eigen_vector = eigen_vector[:, ind]
print(eigen_vector)
print(eigen_value)
np.asarray(eigen_vector)
np.asarray(eigen_vector).T
pc_matrix = np.dot(np.asarray(df), np.asarray(eigen_vector).T)
print(pc_matrix)
print(pc_matrix.shape)
pc_matrix[:, 1].shape
pc1 = pc_matrix[:, 0]
pc2 = pc_matrix[:, 1]
print(pc1.shape)
print(pc2.shape)
var1 = np.var(pc1)
var2 = np.var(pc2)
print(var1)
print(var2)
print(max(var1, var2))
df['OBESITY'] = pc1
df.head()
data = data.drop('SkinThickness', axis=1)
data = data.drop('BMI', axis=1)
data.head()
data['OBESITY'] = pc1
print(data.head())