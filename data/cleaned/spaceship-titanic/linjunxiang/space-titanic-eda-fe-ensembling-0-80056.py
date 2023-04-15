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
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head(5)
n_null = df_train.isnull().sum() / df_train.shape[0]
fig = px.histogram(x=df_train.columns, y=n_null)
fig.show()
n_unique = df_train.nunique() / df_train.shape[0]
fig = px.histogram(x=df_train.columns, y=n_unique)
fig.show()

def corr_matrix_plot(df):
    fig = px.imshow(df.corr(), color_continuous_scale='RdBu_r', origin='lower', text_auto=True, aspect='auto', color_continuous_midpoint=0.0)
    fig.update_layout(font_size=15)
    fig.update_layout(title_text='Correlation Heatmap', plot_bgcolor='rgb(242, 242, 242)', paper_bgcolor='rgb(242, 242, 242)', title_font=dict(size=29, family='Lato, sans-serif'), margin=dict(t=90))
    fig.show()
corr_matrix_plot(df_train)
print(df_train.nunique())
fig = px.histogram(df_train, x='Transported', color='HomePlanet', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs HomePlanet')
fig.show()
fig = px.histogram(df_train, x='Transported', color='CryoSleep', barmode='group')
fig.update_layout(title='Transported vs CryoSleep')
fig.show()
fig = px.histogram(df_train, x='Transported', color='Destination', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs Destination')
fig.show()
fig = px.histogram(df_train, x='Transported', color='VIP', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs VIP')
fig.show()
df_train.info()
numerical_features = df_train.select_dtypes('float').columns
(row, col) = (2, 3)
fig = make_subplots(rows=row, cols=col, subplot_titles=numerical_features)

def grid(*args):
    return np.stack(np.meshgrid(*args, indexing='ij'), axis=-1)
grid_layout = grid(np.arange(1, row + 1), np.arange(1, col + 1)).reshape(row * col, 2)
n = 0
for col in numerical_features:
    fig.add_trace(go.Histogram(x=df_train[col], name=col), row=grid_layout[n][0], col=grid_layout[n][1])
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
df_train['Age'] = df_train['Age'].fillna(df_train.groupby('Transported')['Age'].transform('mean'))
df_train['Name'] = df_train['Name'].fillna('None')
df_test['Name'] = df_test['Name'].fillna('None')
df_train['Cabin'] = df_train['Cabin'].fillna(method='ffill')
df_test['Cabin'] = df_test['Cabin'].fillna(method='ffill')

def split_Cabin(df):
    df['Cabin_first'] = df['Cabin'].apply(lambda x: x.split('/')[0])
    df['Cabin_mid'] = df['Cabin'].apply(lambda x: x.split('/')[1])
    df['Cabin_last'] = df['Cabin'].apply(lambda x: x.split('/')[2])
    df.pop('Cabin')
    return df
split_Cabin(df_train)
split_Cabin(df_test)
fig = px.histogram(df_train, x='Transported', color='Cabin_first', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs Cabin_first')
fig.show()
fig = px.histogram(df_train, x='Transported', color='Cabin_mid', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs Cabin_mid')
fig.show()
fig = px.histogram(df_train, x='Transported', color='Cabin_last', barmode='group', histnorm='percent')
fig.update_layout(title='Transported vs Cabin_last')
fig.show()
df_train.drop('Cabin_mid', axis=1, inplace=True)
df_test.drop('Cabin_mid', axis=1, inplace=True)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_features = df_train.select_dtypes(numerics).columns
categorical_features = df_train.select_dtypes(exclude=numerics).columns.to_list()
categorical_features.remove('Transported')
imputation(df_train, categorical_features, 'categorical')
imputation(df_test, categorical_features, 'categorical')
imputation(df_train, numerical_features, 'numerical')
imputation(df_test, numerical_features, 'numerical')
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