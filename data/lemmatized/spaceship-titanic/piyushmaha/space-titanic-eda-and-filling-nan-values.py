import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from collections import Counter
from colorama import Fore, Back, Style
y_ = Fore.YELLOW
r_ = Fore.RED
g_ = Fore.GREEN
b_ = Fore.BLUE
m_ = Fore.MAGENTA
bl_ = Fore.BLACK
plt.rcParams['figure.figsize'] = (10, 6)
custom_color = ['#17202a', '#e03232', '#b60337', '#fa3f75', '#fa0c40', '#f4d03f']
customPalette = sns.set_palette(sns.color_palette(custom_color))
sns.palplot(sns.color_palette(custom_color), size=1)
YELLOW = sns.dark_palette(custom_color[4], reverse=False)
PINK = sns.dark_palette(custom_color[3])
sns.color_palette(PINK)
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head(5)
_input1.shape
print(f'{y_} Data types of data columns: \n{m_}{_input1.dtypes}')
print(f'{r_}\nOccurance of Dtypes')
[print(f'{g_}{values} Features/columns have dtype {keys}') for (keys, values) in Counter(_input1.dtypes).items()]
_input1.head(2)
_input1.VIP.value_counts()
_input1['VIP'] = _input1[['VIP']].astype(bool)
Na_df = pd.DataFrame(_input1.isnull().sum().sort_values(), columns=['counts'])
Na_df['percentage'] = Na_df['counts'] / 8693 * 100
Na_df
fig = px.imshow(_input1.isnull(), height=480)
fig.show()
train_df_not_null = _input1.dropna(axis=0).reset_index(drop=True)
print(f'{m_}Shape of DataFrame after removing Null values is {train_df_not_null.shape}')
print(f'{g_}Hence the Data is reduced by {(_input1.shape[0] - train_df_not_null.shape[0]) * 100 / 8693}')
df_null = _input1[_input1.isnull().any(axis=1)]
df_null.head(10)
train_df_1 = _input1.copy()
cryo_true = train_df_not_null[train_df_not_null.CryoSleep == True]
cryo_true.head(5)
print(f'{b_}The Observations having CryoSleep True is {cryo_true.shape[0]}')
[print(f'\n{bl_}{name} column have {y_}min value {value.min()} {bl_}and max value {y_}{value.max()}') for (name, value) in cryo_true.iteritems() if value.dtype != object]
sns.countplot(x=cryo_true.VIP)
px.histogram(data_frame=cryo_true, x='Destination', color='HomePlanet')
cryo_false = train_df_not_null[train_df_not_null.CryoSleep == False]
cryo_false.head(8)
px.box(data_frame=cryo_false, x='RoomService')
cryo_false_rates_zero = cryo_false.loc[(cryo_false.RoomService == 0) & (cryo_false.FoodCourt == 0) & (cryo_false.ShoppingMall == 0) & (cryo_false.Spa == 0) & (cryo_false.VRDeck == 0)]
cryo_false_rates_zero.head(10)
cryo_false_rates_zero.shape
plt.subplot(1, 2, 1)
sns.countplot(data=cryo_false_rates_zero, x='HomePlanet')
plt.subplot(1, 2, 2)
sns.countplot(data=cryo_false_rates_zero, x='Destination')
px.histogram(data_frame=cryo_false_rates_zero, x='Destination', color='HomePlanet')

def Plots(df, x, y=None, title=None):
    fig = make_subplots(rows=3, cols=1, row_heights=[0.25, 0.25, 0.5])
    fig.append_trace(go.Box(x=df[x]), row=1, col=1)
    fig.append_trace(go.Violin(x=df[x]), row=2, col=1)
    fig.append_trace(go.Histogram(x=df[x]), row=3, col=1)
    fig.update_xaxes(title_text='Box plot', row=1, col=1)
    fig.update_xaxes(title_text='Voilin plot', row=2, col=1)
    fig.update_xaxes(title_text='Histogram', row=3, col=1)
    fig.update_layout(title_text=title, height=1200, width=1000, showlegend=False)
    fig.show()
train_df_1.isnull().sum()
null_before = train_df_1[train_df_1.CryoSleep == True].isnull().sum().sum()
(train_df_1[train_df_1.CryoSleep == True].isnull().sum(), null_before)
null_before_false = train_df_1[train_df_1.CryoSleep == False].isnull().sum().sum()
(train_df_1[train_df_1.CryoSleep == False].isnull().sum().sum(), train_df_1[train_df_1.CryoSleep == False].isnull().sum())
train_df_1.loc[train_df_1.CryoSleep == True, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = 0
null_after = train_df_1[train_df_1.CryoSleep == True].isnull().sum().sum()
(train_df_1[train_df_1.CryoSleep == True].isnull().sum(), null_after)
print(f'The Null values are reduced by {(null_before - null_after) * 100 / null_before}')
fig = go.Figure()
fig.add_trace(go.Box(y=cryo_false.RoomService, name='Room Service', boxmean=True))
fig.add_trace(go.Box(y=cryo_false.FoodCourt, name='Food Court', boxmean=True))
fig.add_trace(go.Box(y=cryo_false.ShoppingMall, name='Shopping mall', boxmean=True))
fig.add_trace(go.Box(y=cryo_false.Spa, name='Spa', boxmean=True))
fig.add_trace(go.Box(y=cryo_false.VRDeck, name='VRDeck', boxmean=True))
fig = go.Figure()
fig.add_trace(go.Violin(y=cryo_false.RoomService, name='Room Service'))
fig.add_trace(go.Violin(y=cryo_false.FoodCourt, name='Food Court'))
fig.add_trace(go.Violin(y=cryo_false.ShoppingMall, name='Shopping mall'))
fig.add_trace(go.Violin(y=cryo_false.Spa, name='Spa'))
fig.add_trace(go.Violin(y=cryo_false.VRDeck, name='VRDeck'))
cryo_false.median(numeric_only=True)
train_df_1 = train_df_1.fillna(cryo_false.median(numeric_only=True))
train_df_1.isnull().sum()
null_after_false = train_df_1[train_df_1.CryoSleep == False].isnull().sum().sum()
print(f'The reduction in null values when CryoSleep is False is {(null_before_false - null_after_false) * 100 / null_before_false}')
print(f'{y_}Total Reduction in null values is: ')
total_null = _input1.isnull().sum().sum()
total_null_after = train_df_1.isnull().sum().sum()
print(f'{b_}{(total_null - total_null_after) * 100 / total_null}')
train_df_1 = train_df_1.drop('Name', axis=1, inplace=False)
train_df_1.isnull().sum()
train_df_1.shape
a = round((train_df_1.shape[0] - train_df_1.dropna(axis=0).shape[0]) * 100 / train_df_1.shape[0], 4)
print(f'{bl_} After Dropping Null values the dataset is reduced by {b_}{a}')
train_df_1 = train_df_1.dropna(axis=0, inplace=False)

def Plots(df, x, y=None, title=None):
    fig = make_subplots(rows=3, cols=1, row_heights=[0.25, 0.25, 0.5])
    fig.append_trace(go.Box(x=df[x]), row=1, col=1)
    fig.append_trace(go.Violin(x=df[x]), row=2, col=1)
    fig.append_trace(go.Histogram(x=df[x]), row=3, col=1)
    fig.update_xaxes(title_text='Box plot', row=1, col=1)
    fig.update_xaxes(title_text='Voilin plot', row=2, col=1)
    fig.update_xaxes(title_text='Histogram', row=3, col=1)
    fig.update_layout(title_text=title, height=1200, width=1000, showlegend=False)
    fig.show()

def Pie_plots(data_frame, feature_name):
    label = data_frame[feature_name].value_counts().index
    values = data_frame[feature_name].value_counts().values
    fig = go.Figure(data=[go.Pie(labels=label, values=values, hole=0.3, pull=[0.025] * len(label))])
    fig.show()

def Scatter_plot(data_frame, feature1, feature2, count_plot=None, size=None, color=None):
    fig = px.scatter(data_frame=data_frame, x=feature1, y=feature2, color=color, size=size, marginal_x='histogram', marginal_y='violin')
    fig.update_layout(width=1000, height=800)
    fig.show()
Scatter_plot(train_df_1, 'RoomService', 'FoodCourt', color='CryoSleep')
Plots(train_df_1, x='RoomService', title='Room Service rates')
Pie_plots(train_df_1, 'HomePlanet')
Pie_plots(train_df_1, 'CryoSleep')