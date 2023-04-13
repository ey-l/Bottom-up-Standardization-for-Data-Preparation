import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
diabetes_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_df.info()
diabetes_df.shape
diabetes_df.head(10)
diabetes_df.describe()
diabetes_df.isnull().sum()
diabetes_df.nunique()
diabetes_df.groupby(diabetes_df['Outcome']).count()
pass
var_list = diabetes_df.columns.unique()
for (index, column) in enumerate(var_list):
    pass
    pass
    pass
pass
fill_list = ['SkinThickness', 'Insulin', 'BMI', 'BloodPressure']
for i in fill_list:
    diabetes_df[i].replace(0, diabetes_df[i].mean(), inplace=True)
diabetes_df.describe()
pass
var_list = diabetes_df.columns.unique()
for (index, column) in enumerate(var_list):
    pass
    pass
    pass
pass
print(stats.describe(diabetes_df['Glucose']))
print(stats.describe(diabetes_df['BloodPressure']))
pass
pass
pass
pass
corr_with_outcome = diabetes_df.corrwith(diabetes_df['Outcome']).sort_values(ascending=False)
print(corr_with_outcome)
diabetes_df.var()
diabetes_df = diabetes_df.dropna()
X = diabetes_df[['Glucose', 'Insulin', 'Age', 'SkinThickness', 'BMI', 'Pregnancies']]
X = StandardScaler().fit_transform(diabetes_df)
Xt = X.T
Cx = np.cov(Xt)
print('Covariance Matrix:\n', Cx)
(eig_val_cov, eig_vec_cov) = np.linalg.eig(Cx)
for i in range(len(eig_val_cov)):
    eigvec_cov = eig_vec_cov[:, i].T
    print('Eigenvector {}: \n{}'.format(i + 1, eigvec_cov))
    print('Eigenvalue {}: {}'.format(i + 1, eig_val_cov[i]))
    print(40 * '-')
    print('Proportion of total variance', eig_val_cov / sum(eig_val_cov))
pass
pass
print(eig_val_cov)
X = diabetes_df
X = StandardScaler().fit_transform(X)
sklearn_pca = PCA(n_components=1)
diabetes_df['pca1'] = sklearn_pca.fit_transform(X)
print('percentage of total variance in data set explained by each component in Sklearn PCA.', sklearn_pca.explained_variance_ratio_)
diabetes_df.corrwith(diabetes_df['Outcome']).sort_values(ascending=False)