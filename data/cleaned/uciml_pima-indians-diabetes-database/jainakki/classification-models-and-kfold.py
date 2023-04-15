import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.sample(n=10, random_state=0)
data.info(verbose=True)
data.describe().transpose()
import seaborn as sns
sns.pairplot(data, hue='Outcome', diag_kind='kde')

data_copy = data.copy(deep=True)
data_copy.loc[:, ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data_copy.loc[:, ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
data_copy.isnull().sum()
import missingno as msno
msno.matrix(data_copy, figsize=(20, 10), labels=True, color=(0.502, 0.0, 0.0))

import plotly.graph_objects as go
column_names = data_copy.columns
no_of_boxes = len(column_names)
colors = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, no_of_boxes)]
fig = go.Figure(data=[go.Box(y=data_copy.loc[:, column_names[i]], marker_color=colors[i], name=column_names[i], boxmean=True, showlegend=True) for i in range(no_of_boxes)])
fig.update_layout(xaxis=dict(showgrid=True, zeroline=True, showticklabels=True), yaxis=dict(zeroline=True, gridcolor='white'), paper_bgcolor='rgb(233,233,233)', plot_bgcolor='rgb(233,233,233)')
fig.show()
from sklearn.impute import SimpleImputer
imputer_mean = SimpleImputer(missing_values=np.NaN, strategy='mean')
imputer_median = SimpleImputer(missing_values=np.NaN, strategy='median')