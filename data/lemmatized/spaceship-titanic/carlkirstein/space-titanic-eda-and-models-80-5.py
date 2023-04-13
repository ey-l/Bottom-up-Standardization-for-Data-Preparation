import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib
import warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)
reload(plt)
warnings.filterwarnings('ignore')
pio.renderers.default = 'iframe'
pio.templates['ck_template'] = go.layout.Template(layout_autosize=False, layout_width=800, layout_height=600, layout_font=dict(family='Calibri Light'), layout_title_font=dict(family='Calibri'), layout_hoverlabel_font=dict(family='Calibri Light'))
pio.templates.default = 'ck_template+gridon'
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.info()
list(_input1)
_input1.head(10)
_input1.describe(include='all')
plt.figure(figsize=(15, 15))
threshold = 0.0
sns.set_style('whitegrid', {'axes.facecolor': '.0'})
df_cluster2 = _input1.corr()
mask = df_cluster2.where(abs(df_cluster2) >= threshold).isna()
plot_kws = {'s': 1}
sns.heatmap(df_cluster2, cmap='RdYlBu', annot=True, mask=mask, linewidths=0.2, linecolor='lightgrey').set_facecolor('white')
from plotly.subplots import make_subplots
norm_width = 1.5
high_width = 2.5
title_set = []
secondary_Y = []
for feature in _input1.columns:
    title_set.append(feature)
fig = make_subplots(rows=len(_input1.columns), cols=1, subplot_titles=title_set)
fig.update_layout(title='Comparison of Labels', height=400 * len(_input1.columns), showlegend=False)
i = 0
for feature in _input1.columns:
    i += 1
    x0 = _input1[feature][_input1['Transported'] == 0]
    x1 = _input1[feature][_input1['Transported'] == 1]
    hist_data = [x0, x1]
    group_labels = ['Label 0', 'Label 1']
    try:
        fig.add_trace(go.Violin(y=x0, x=_input1['Transported'], jitter=0.3, meanline_visible=True), row=i, col=1, secondary_y=False)
    except:
        pass
fig.show()
_input1.describe(include='all')
from pandas_profiling import ProfileReport
_input1['PassengerId'].str[0:4].value_counts()
_input1['PassengerId'].str[-2:].value_counts()
len(_input1[_input1['HomePlanet'].isna()])
_input1['HomePlanet'].value_counts()
df_group = _input1.groupby(['HomePlanet', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='HomePlanet', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
_input1.groupby('HomePlanet')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
len(_input1[_input1['CryoSleep'].isna()])
_input1['CryoSleep'].value_counts()
df_group = _input1.groupby(['CryoSleep', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='CryoSleep', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
_input1.groupby('CryoSleep')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
_input1['Cabin'].value_counts()
fig = px.bar(pd.DataFrame(_input1['Cabin'].str[0].value_counts()).reset_index(), x='Cabin', y='index', orientation='h')
fig.show()
fig = px.bar(pd.DataFrame(_input1['Cabin'].str[-1].value_counts()).reset_index(), x='Cabin', y='index', orientation='h')
fig.show()
_input1['Cabin Deck'] = _input1['Cabin'].str[0]
_input1['Cabin Side'] = _input1['Cabin'].str[-1]
df_group = _input1.groupby(['Cabin Deck', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='Cabin Deck', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
_input1.groupby('Cabin Deck')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
df_group = _input1.groupby(['Cabin Side', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='Cabin Side', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
_input1.groupby('Cabin Side')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
fig = px.bar(pd.DataFrame(_input1['Destination'].value_counts()).reset_index(), x='Destination', y='index', orientation='h')
fig.show()
df_group = _input1.groupby(['Destination', 'Transported'])
df_group.count().reset_index()
df_group = _input1.groupby(['Destination', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='Destination', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
_input1.groupby('HomePlanet')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
df_data = _input1
temp = df_data[df_data.Age.isnull() == False][['Transported', 'Age']]
temp['Transported'] = temp['Transported'].replace([False, True], [0, 1], inplace=False)
temp['Transported'] = temp.groupby('Age')['Transported'].transform('mean')
fig = px.scatter(temp, x='Age', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability by Age', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
df_group = _input1.groupby(['VIP', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='VIP', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
_input1.groupby('VIP')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
_input1['RoomService'].value_counts()
temp = df_data[df_data.Age.isnull() == False][['Transported', 'RoomService']]
temp['Transported'] = temp['Transported'].replace([False, True], [0, 1], inplace=False)
temp['Transported'] = temp.groupby('RoomService')['Transported'].transform('mean')
fig = px.scatter(temp, x='RoomService', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability by RoomService', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
_input1['FoodCourt'].value_counts()
temp = df_data[df_data.Age.isnull() == False][['Transported', 'FoodCourt']]
temp['Transported'] = temp['Transported'].replace([False, True], [0, 1], inplace=False)
temp['Transported'] = temp.groupby('FoodCourt')['Transported'].transform('mean')
fig = px.scatter(temp, x='FoodCourt', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
_input1['ShoppingMall'].value_counts()
temp = df_data[df_data.Age.isnull() == False][['Transported', 'ShoppingMall']]
temp['Transported'] = temp['Transported'].replace([False, True], [0, 1], inplace=False)
temp['Transported'] = temp.groupby('ShoppingMall')['Transported'].transform('mean')
fig = px.scatter(temp, x='ShoppingMall', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
_input1['Spa'].value_counts()
temp = df_data[df_data.Age.isnull() == False][['Transported', 'Spa']]
temp['Transported'] = temp['Transported'].replace([False, True], [0, 1], inplace=False)
temp['Transported'] = temp.groupby('Spa')['Transported'].transform('mean')
fig = px.scatter(temp, x='Spa', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
_input1['VRDeck'].value_counts()
temp = df_data[df_data.Age.isnull() == False][['Transported', 'VRDeck']]
temp['Transported'] = temp['Transported'].replace([False, True], [0, 1], inplace=False)
temp['Transported'] = temp.groupby('VRDeck')['Transported'].transform('mean')
fig = px.scatter(temp, x='VRDeck', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
_input1['Name'].value_counts()
_input1[['FirstName', 'Surname']] = _input1['Name'].str.split(' ', expand=True)
df_group = _input1.groupby(['FirstName', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='FirstName', color='Transported', orientation='h')
fig.update_layout(barmode='relative')
fig.show()
df_group = _input1.groupby(['Surname', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='Surname', color='Transported', orientation='h')
fig.update_layout(barmode='relative')
fig.show()
df_group = _input1.groupby(['Transported'])
df_group.count()
_input1[_input1['Transported'].isna()]
_input1['Target'] = _input1['Transported'] + 1 - 1
_input1['CryoSleep'] = _input1['CryoSleep'] + 1 - 1
_input1['VIP'] = _input1['VIP'] + 1 - 1
_input1
list_drop = ['PassengerId', 'Transported', 'Name', 'FirstName', 'Surname', 'Cabin']
_input1 = _input1.drop(list_drop, axis=1, inplace=False)
df_numeric = _input1.select_dtypes(include=[np.number])
df_numeric.describe(include='all')
df_numeric = df_numeric.fillna(df_numeric.mean(), inplace=False)
df_cat = _input1.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')
df_cat['HomePlanet'].value_counts()
df_cat['CryoSleep'].value_counts()
df_cat['VIP'].value_counts()
_input1 = _input1.dropna(axis=0, how='any', inplace=False)
df_numeric = _input1.select_dtypes(include=[np.number])
df_numeric.describe(include='all')
DEBUG = 0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('max = ' + str(df_numeric[feature].max()))
        print('75th = ' + str(df_numeric[feature].quantile(0.75)))
        print('median = ' + str(df_numeric[feature].median()))
        print(df_numeric[feature].max - _input1[feature].quantile(0.75) > _input1[feature].quantile(0.75) - df_numeric[feature].median())
        print('----------------------------------------------------')
    if _input1[feature].max() - _input1[feature].quantile(0.75) > _input1[feature].quantile(0.75) - _input1[feature].median() and _input1[feature].max() > 10:
        _input1[feature] = np.where(_input1[feature] < _input1[feature].quantile(0.99), _input1[feature], _input1[feature].quantile(0.99))
df_numeric = _input1.select_dtypes(include=[np.number])
df_numeric.describe(include='all')
df_numeric = _input1.select_dtypes(include=[np.number])
df_before = df_numeric.copy()
DEBUG = 0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = ' + str(df_numeric[feature].nunique()))
        print(df_numeric[feature].nunique() > 50)
        print('----------------------------------------------------')
    if df_numeric[feature].nunique() > 50:
        if df_numeric[feature].min() == 0:
            _input1[feature] = np.log(_input1[feature] + 1)
        else:
            _input1[feature] = np.log(_input1[feature])
df_numeric = _input1.select_dtypes(include=[np.number])
_input1 = pd.get_dummies(_input1, drop_first=True)
df_numeric = _input1.select_dtypes(include=[np.number])
df_cat = _input1.select_dtypes(exclude=[np.number])
X = _input1.drop('Target', axis=1)
y = _input1['Target']
feature_names = list(X.columns)
_input1.head(10)
from sklearn.feature_selection import SelectKBest, chi2
best_features = SelectKBest(score_func=chi2, k='all')
X = _input1.drop('Target', axis=1)
y = _input1['Target']