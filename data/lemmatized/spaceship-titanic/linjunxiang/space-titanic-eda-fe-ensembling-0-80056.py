import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head(5)
n_null = _input1.isnull().sum() / _input1.shape[0]
fig = px.histogram(x=_input1.columns, y=n_null)
fig.show()
n_unique = _input1.nunique() / _input1.shape[0]
fig = px.histogram(x=_input1.columns, y=n_unique)
fig.show()

def corr_matrix_plot(df):
    fig = px.imshow(df.corr(), color_continuous_scale='RdBu_r', origin='lower', text_auto=True, aspect='auto', color_continuous_midpoint=0.0)
    fig.update_layout(font_size=15)
    fig.update_layout(title_text='Correlation Heatmap', plot_bgcolor='rgb(242, 242, 242)', paper_bgcolor='rgb(242, 242, 242)', title_font=dict(size=29, family='Lato, sans-serif'), margin=dict(t=90))
    fig.show()
corr_matrix_plot(_input1)
print(_input1.nunique())
fig = px.histogram(_input1, x='Transported', color='HomePlanet', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs HomePlanet')
fig.show()
fig = px.histogram(_input1, x='Transported', color='CryoSleep', barmode='group')
fig.update_layout(title='Transported vs CryoSleep')
fig.show()
fig = px.histogram(_input1, x='Transported', color='Destination', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs Destination')
fig.show()
fig = px.histogram(_input1, x='Transported', color='VIP', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs VIP')
fig.show()
_input1.info()
numerical_features = _input1.select_dtypes('float').columns
(row, col) = (2, 3)
fig = make_subplots(rows=row, cols=col, subplot_titles=numerical_features)

def grid(*args):
    return np.stack(np.meshgrid(*args, indexing='ij'), axis=-1)
grid_layout = grid(np.arange(1, row + 1), np.arange(1, col + 1)).reshape(row * col, 2)
n = 0
for col in numerical_features:
    fig.add_trace(go.Histogram(x=_input1[col], name=col), row=grid_layout[n][0], col=grid_layout[n][1])
    n += 1
fig.update_layout(autosize=False, width=800, height=600)
fig.show()

def imputation(df, features, type_):
    """
    General imputator through "mode" or "median"
    """
    if type_ == 'categorical':
        for col in features:
            print(f'{col} fill by {df[col].mode()[0]}')
            df[col] = df[col].fillna(df[col].mode()[0])
    else:
        for col in features:
            print(f'{col} fill by {df[col].median()}')
            df[col] = df[col].fillna(df[col].median())
_input1['Age'] = _input1['Age'].fillna(_input1.groupby('Transported')['Age'].transform('mean'))
_input1['Name'] = _input1['Name'].fillna('None')
_input0['Name'] = _input0['Name'].fillna('None')
_input1['Cabin'] = _input1['Cabin'].fillna(method='ffill')
_input0['Cabin'] = _input0['Cabin'].fillna(method='ffill')

def split_Cabin(df):
    df['Cabin_first'] = df['Cabin'].apply(lambda x: x.split('/')[0])
    df['Cabin_mid'] = df['Cabin'].apply(lambda x: x.split('/')[1])
    df['Cabin_last'] = df['Cabin'].apply(lambda x: x.split('/')[2])
    df.pop('Cabin')
    return df
split_Cabin(_input1)
split_Cabin(_input0)
fig = px.histogram(_input1, x='Transported', color='Cabin_first', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs Cabin_first')
fig.show()
fig = px.histogram(_input1, x='Transported', color='Cabin_mid', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs Cabin_mid')
fig.show()
fig = px.histogram(_input1, x='Transported', color='Cabin_last', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs Cabin_last')
fig.show()
_input1 = _input1.drop('Cabin_mid', axis=1, inplace=False)
_input0 = _input0.drop('Cabin_mid', axis=1, inplace=False)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_features = _input1.select_dtypes(numerics).columns
categorical_features = _input1.select_dtypes(exclude=numerics).columns.to_list()
categorical_features.remove('Transported')
imputation(_input1, categorical_features, 'categorical')
imputation(_input0, categorical_features, 'categorical')
imputation(_input1, numerical_features, 'numerical')
imputation(_input0, numerical_features, 'numerical')
outliers = []

def z_score_detector(df):
    mean_ = df.mean()
    std = df.std()
    for (i, value) in enumerate(df):
        z = (value - mean_) / std
        if abs(z) > 3:
            outliers.append(i)
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def dbscan(df, feature, n):
    cols = ['Transported', feature]
    X = StandardScaler().fit_transform(df[df['Transported'].isnull() == False][cols].copy().values)