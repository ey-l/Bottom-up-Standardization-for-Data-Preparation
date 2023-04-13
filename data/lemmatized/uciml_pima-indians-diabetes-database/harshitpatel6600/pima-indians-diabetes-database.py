import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
columns = df.columns.tolist()
columns
df.isna().sum()
pass
corr_mat = df.corr()
corr_mat = corr_mat.to_numpy()
corr_mat = np.round(corr_mat, 2)
import plotly.figure_factory as ff
z = corr_mat
fig = ff.create_annotated_heatmap(z, colorscale='Viridis', x=columns, y=columns, showscale=True)
fig['layout']['xaxis']['side'] = 'bottom'
pass
positive = len(df.loc[df['Outcome'] == 1])
negative = len(df.loc[df['Outcome'] == 0])
print('Number of Positive result: ' + str(positive))
print('Number of Negative result: ' + str(negative))
colors = ['lightslategray'] * 2
colors[1] = 'crimson'
fig = go.Figure(go.Bar(x=['Positive', 'Negative'], y=[positive, negative], text=[positive, negative], textposition='auto', marker_color=colors))
fig.update_layout(title_text='Number of Positive and Negative cases')
pass
from sklearn.model_selection import train_test_split
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sc_X = pd.DataFrame(scaler.fit_transform(X.copy()))
(X_train, X_test, y_train, y_test) = train_test_split(sc_X, y, test_size=0.33, random_state=1)
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
for i in range(1, 15):
    knn = KNeighborsClassifier(i)