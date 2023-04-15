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
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.info()
list(df)
df.head(10)
df.describe(include='all')
plt.figure(figsize=(15, 15))
threshold = 0.0
sns.set_style('whitegrid', {'axes.facecolor': '.0'})
df_cluster2 = df.corr()
mask = df_cluster2.where(abs(df_cluster2) >= threshold).isna()
plot_kws = {'s': 1}
sns.heatmap(df_cluster2, cmap='RdYlBu', annot=True, mask=mask, linewidths=0.2, linecolor='lightgrey').set_facecolor('white')
from plotly.subplots import make_subplots
norm_width = 1.5
high_width = 2.5
title_set = []
secondary_Y = []
for feature in df.columns:
    title_set.append(feature)
fig = make_subplots(rows=len(df.columns), cols=1, subplot_titles=title_set)
fig.update_layout(title='Comparison of Labels', height=400 * len(df.columns), showlegend=False)
i = 0
for feature in df.columns:
    i += 1
    x0 = df[feature][df['Transported'] == 0]
    x1 = df[feature][df['Transported'] == 1]
    hist_data = [x0, x1]
    group_labels = ['Label 0', 'Label 1']
    try:
        fig.add_trace(go.Violin(y=x0, x=df['Transported'], jitter=0.3, meanline_visible=True), row=i, col=1, secondary_y=False)
    except:
        pass
fig.show()
df.describe(include='all')

from pandas_profiling import ProfileReport

df['PassengerId'].str[0:4].value_counts()
df['PassengerId'].str[-2:].value_counts()
len(df[df['HomePlanet'].isna()])
df['HomePlanet'].value_counts()
df_group = df.groupby(['HomePlanet', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='HomePlanet', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
df.groupby('HomePlanet')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
len(df[df['CryoSleep'].isna()])
df['CryoSleep'].value_counts()
df_group = df.groupby(['CryoSleep', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='CryoSleep', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
df.groupby('CryoSleep')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
df['Cabin'].value_counts()
fig = px.bar(pd.DataFrame(df['Cabin'].str[0].value_counts()).reset_index(), x='Cabin', y='index', orientation='h')
fig.show()
fig = px.bar(pd.DataFrame(df['Cabin'].str[-1].value_counts()).reset_index(), x='Cabin', y='index', orientation='h')
fig.show()
df['Cabin Deck'] = df['Cabin'].str[0]
df['Cabin Side'] = df['Cabin'].str[-1]
df_group = df.groupby(['Cabin Deck', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='Cabin Deck', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
df.groupby('Cabin Deck')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
df_group = df.groupby(['Cabin Side', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='Cabin Side', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
df.groupby('Cabin Side')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
fig = px.bar(pd.DataFrame(df['Destination'].value_counts()).reset_index(), x='Destination', y='index', orientation='h')
fig.show()
df_group = df.groupby(['Destination', 'Transported'])
df_group.count().reset_index()
df_group = df.groupby(['Destination', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='Destination', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
df.groupby('HomePlanet')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
df_data = df
temp = df_data[df_data.Age.isnull() == False][['Transported', 'Age']]
temp['Transported'].replace([False, True], [0, 1], inplace=True)
temp['Transported'] = temp.groupby('Age')['Transported'].transform('mean')
fig = px.scatter(temp, x='Age', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability by Age', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
df_group = df.groupby(['VIP', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='VIP', color='Transported', orientation='h')
fig.update_layout(barmode='group')
fig.show()
df.groupby('VIP')['Transported'].value_counts(normalize=True).unstack('Transported').plot.barh(stacked=True, figsize=(10, 10))
df['RoomService'].value_counts()
temp = df_data[df_data.Age.isnull() == False][['Transported', 'RoomService']]
temp['Transported'].replace([False, True], [0, 1], inplace=True)
temp['Transported'] = temp.groupby('RoomService')['Transported'].transform('mean')
fig = px.scatter(temp, x='RoomService', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability by RoomService', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
df['FoodCourt'].value_counts()
temp = df_data[df_data.Age.isnull() == False][['Transported', 'FoodCourt']]
temp['Transported'].replace([False, True], [0, 1], inplace=True)
temp['Transported'] = temp.groupby('FoodCourt')['Transported'].transform('mean')
fig = px.scatter(temp, x='FoodCourt', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
df['ShoppingMall'].value_counts()
temp = df_data[df_data.Age.isnull() == False][['Transported', 'ShoppingMall']]
temp['Transported'].replace([False, True], [0, 1], inplace=True)
temp['Transported'] = temp.groupby('ShoppingMall')['Transported'].transform('mean')
fig = px.scatter(temp, x='ShoppingMall', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
df['Spa'].value_counts()
temp = df_data[df_data.Age.isnull() == False][['Transported', 'Spa']]
temp['Transported'].replace([False, True], [0, 1], inplace=True)
temp['Transported'] = temp.groupby('Spa')['Transported'].transform('mean')
fig = px.scatter(temp, x='Spa', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
df['VRDeck'].value_counts()
temp = df_data[df_data.Age.isnull() == False][['Transported', 'VRDeck']]
temp['Transported'].replace([False, True], [0, 1], inplace=True)
temp['Transported'] = temp.groupby('VRDeck')['Transported'].transform('mean')
fig = px.scatter(temp, x='VRDeck', y='Transported')
fig.update_xaxes(showgrid=False, showline=True, gridwidth=0.05, linecolor='gray', zeroline=False, linewidth=2)
fig.update_yaxes(showline=True, gridwidth=0.05, linecolor='gray', linewidth=2, zeroline=False)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig.update_layout(height=600, margin=dict(b=50, r=30, l=100, t=100), title='Transported Probability', hoverlabel=dict(font_color='floralwhite'), showlegend=False)
fig.show()
df['Name'].value_counts()
df[['FirstName', 'Surname']] = df['Name'].str.split(' ', expand=True)
df_group = df.groupby(['FirstName', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='FirstName', color='Transported', orientation='h')
fig.update_layout(barmode='relative')
fig.show()
df_group = df.groupby(['Surname', 'Transported'])
df_group.count().reset_index()
fig = px.bar(df_group.count().reset_index(), x='PassengerId', y='Surname', color='Transported', orientation='h')
fig.update_layout(barmode='relative')
fig.show()
df_group = df.groupby(['Transported'])
df_group.count()
df[df['Transported'].isna()]
df['Target'] = df['Transported'] + 1 - 1
df['CryoSleep'] = df['CryoSleep'] + 1 - 1
df['VIP'] = df['VIP'] + 1 - 1
df
list_drop = ['PassengerId', 'Transported', 'Name', 'FirstName', 'Surname', 'Cabin']
df.drop(list_drop, axis=1, inplace=True)
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all')
df_numeric.fillna(df_numeric.mean(), inplace=True)
df_cat = df.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')
df_cat['HomePlanet'].value_counts()
df_cat['CryoSleep'].value_counts()
df_cat['VIP'].value_counts()
df.dropna(axis=0, how='any', inplace=True)
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all')
DEBUG = 0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('max = ' + str(df_numeric[feature].max()))
        print('75th = ' + str(df_numeric[feature].quantile(0.75)))
        print('median = ' + str(df_numeric[feature].median()))
        print(df_numeric[feature].max - df[feature].quantile(0.75) > df[feature].quantile(0.75) - df_numeric[feature].median())
        print('----------------------------------------------------')
    if df[feature].max() - df[feature].quantile(0.75) > df[feature].quantile(0.75) - df[feature].median() and df[feature].max() > 10:
        df[feature] = np.where(df[feature] < df[feature].quantile(0.99), df[feature], df[feature].quantile(0.99))
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all')
df_numeric = df.select_dtypes(include=[np.number])
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
            df[feature] = np.log(df[feature] + 1)
        else:
            df[feature] = np.log(df[feature])
df_numeric = df.select_dtypes(include=[np.number])
df = pd.get_dummies(df, drop_first=True)
df_numeric = df.select_dtypes(include=[np.number])
df_cat = df.select_dtypes(exclude=[np.number])
X = df.drop('Target', axis=1)
y = df['Target']
feature_names = list(X.columns)
df.head(10)
from sklearn.feature_selection import SelectKBest, chi2
best_features = SelectKBest(score_func=chi2, k='all')
X = df.drop('Target', axis=1)
y = df['Target']