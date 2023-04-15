import numpy as np
import pandas as pd
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
train = pd.read_csv('data/input/digit-recognizer/train.csv')
train.head()
print(train.shape)
target = train['label']
train = train.drop('label', axis=1)
from sklearn.preprocessing import StandardScaler
X = train.values
X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
(eig_vals, eig_vecs) = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
tot = sum(eig_vals)
var_exp = [i / tot * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
trace1 = go.Scatter(x=list(range(784)), y=cum_var_exp, mode='lines+markers', name="'Cumulative Explained Variance'", line=dict(shape='spline', color='goldenrod'))
trace2 = go.Scatter(x=list(range(784)), y=var_exp, mode='lines+markers', name="'Individual Explained Variance'", line=dict(shape='linear', color='black'))
fig = tls.make_subplots(insets=[{'cell': (1, 1), 'l': 0.7, 'b': 0.5}], print_grid=True)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.layout.title = 'Explained Variance plots - Full and Zoomed-in'
fig.layout.xaxis = dict(range=[0, 80], title='Feature columns')
fig.layout.yaxis = dict(range=[0, 60], title='Explained Variance')
n_components = 30