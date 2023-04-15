import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import catboost
from catboost import Pool
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.info()
train.isnull().sum()
test.info()
test.isnull().sum()
train['CryoSleep'] = train['CryoSleep'].astype(str)
train['VIP'] = train['VIP'].astype(str)
test['CryoSleep'] = test['CryoSleep'].astype(str)
test['VIP'] = test['VIP'].astype(str)
train['Transported'] = train['Transported'].astype(int)
train['CryoSleep'].replace('False', 0, inplace=True)
train['CryoSleep'].replace('True', 1, inplace=True)
train['VIP'].replace('False', 0, inplace=True)
train['VIP'].replace('True', 1, inplace=True)
test['CryoSleep'].replace('False', 0, inplace=True)
test['CryoSleep'].replace('True', 1, inplace=True)
test['VIP'].replace('False', 0, inplace=True)
test['VIP'].replace('True', 1, inplace=True)
train['gggg'] = train['PassengerId'].str.split('_', 1).str[0]
train['pp'] = train['PassengerId'].str.split('_', 1).str[1]
train['firstname'] = train['Name'].str.split(' ', 1).str[0]
train['lastname'] = train['Name'].str.split(' ', 1).str[1]
train['Cabin_desk'] = train['Cabin'].str.split('/', 1).str[0]
train['Cabin_num'] = train['Cabin'].str.split('/', 1).str[1]
train['Cabin_side'] = train['Cabin_num'].str.split('/', 1).str[1]
train['Cabin_num'] = train['Cabin_num'].str.split('/', 1).str[0]
train.drop('Cabin', axis=1, inplace=True)
train.drop('Name', axis=1, inplace=True)
test['gggg'] = test['PassengerId'].str.split('_', 1).str[0]
test['pp'] = test['PassengerId'].str.split('_', 1).str[1]
test['firstname'] = test['Name'].str.split(' ', 1).str[0]
test['lastname'] = test['Name'].str.split(' ', 1).str[1]
test['Cabin_desk'] = test['Cabin'].str.split('/', 1).str[0]
test['Cabin_num'] = test['Cabin'].str.split('/', 1).str[1]
test['Cabin_side'] = test['Cabin_num'].str.split('/', 1).str[1]
test['Cabin_num'] = test['Cabin_num'].str.split('/', 1).str[0]
test.drop('Name', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
train['firstname'].fillna('missing', inplace=True)
train['lastname'].fillna('missing', inplace=True)
for i in range(train.shape[0]):
    if train.iloc[i, 16] == 'missing':
        if train.iloc[i, 13] == train.iloc[i - 1, 13]:
            train.iloc[i, 16] = train.iloc[i - 1, 16]
        elif train.iloc[i, 13] == train.iloc[i + 1, 13]:
            train.iloc[i, 16] = train.iloc[i + 1, 16]
test['firstname'].fillna('missing', inplace=True)
test['lastname'].fillna('missing', inplace=True)
for i in range(test.shape[0]):
    if test.iloc[i, 15] == 'missing':
        if test.iloc[i, 12] == test.iloc[i - 1, 12]:
            test.iloc[i, 15] = test.iloc[i - 1, 15]
        elif test.iloc[i, 12] == test.iloc[i + 1, 12]:
            test.iloc[i, 15] = test.iloc[i + 1, 15]

def ABC(x):
    if x == 'A':
        return 1
    elif x == 'B':
        return 2
    elif x == 'C':
        return 3
    elif x == 'D':
        return 4
    elif x == 'E':
        return 5
    elif x == 'F':
        return 6
    elif x == 'G':
        return 7
    elif x == 'T':
        return 8
train['Cabin_side'].replace('P', 0, inplace=True)
train['Cabin_side'].replace('S', 1, inplace=True)
test['Cabin_side'].replace('P', 0, inplace=True)
test['Cabin_side'].replace('S', 1, inplace=True)
train['Cabin_desk'] = train['Cabin_desk'].map(ABC)
test['Cabin_desk'] = test['Cabin_desk'].map(ABC)
train['VIP'].replace('nan', 0, inplace=True)
test['VIP'].replace('nan', 0, inplace=True)
train['CryoSleep'].replace('nan', 0, inplace=True)
test['CryoSleep'].replace('nan', 0, inplace=True)
for columns in train.columns:
    if train[columns].dtype == object:
        train[columns].fillna(method='bfill', inplace=True)
    else:
        train[columns].fillna(train[columns].mean(), inplace=True)
for columns in test.columns:
    if test[columns].dtype == object:
        test[columns].fillna(method='bfill', inplace=True)
    else:
        test[columns].fillna(test[columns].mean(), inplace=True)
train['Cabin_num'].fillna(method='bfill', inplace=True)
train['Cabin_num'] = train['Cabin_num'].astype(int)
train['Cabin_desk'] = train['Cabin_desk'].astype(int)
train['Cabin_side'] = train['Cabin_side'].astype(int)
test['Cabin_num'].fillna(method='bfill', inplace=True)
test['Cabin_num'] = test['Cabin_num'].astype(int)
test['Cabin_desk'] = test['Cabin_desk'].astype(int)
test['Cabin_side'] = test['Cabin_side'].astype(int)

def LOC(x):
    if x <= 450:
        return 0
    elif 450 < x <= 900:
        return 1
    elif 900 < x <= 1350:
        return 2
    elif x > 1350:
        return 3
train['Cabin_zone'] = train['Cabin_num'].map(LOC)
test['Cabin_zone'] = test['Cabin_num'].map(LOC)
temp = dict(layout=go.Layout(font=dict(family='Franklin Gothic', size=12), height=500, width=1000))
plt.style.use('ggplot')
(f, (ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(75, 25), dpi=100)
ax1.boxplot(x=train['Age'])
ax1.set_title('Overall Distribution', fontsize=30)
ax1.tick_params(labelsize=30)
df = train.loc[train['Transported'] == 0, :]
ax2.boxplot(x=df['Age'])
ax2.set_title('Not Transported Distribution', fontsize=30)
ax2.tick_params(labelsize=30)
df = train.loc[train['Transported'] == 1, :]
ax3.boxplot(x=df['Age'])
ax3.set_title('Transported Distribution', fontsize=30)
ax3.tick_params(labelsize=30)
target = train.Transported.value_counts(normalize=True)
target.rename(index={1: 'True', 0: 'False'}, inplace=True)
(pal, color) = (['aliceblue', 'mistyrose', 'cornsilk', 'honeydew', 'plum'], ['skyblue', 'salmon', 'gold', 'darkseagreen', 'blueviolet'])
fig = go.Figure()
fig.add_trace(go.Pie(labels=target.index, values=target * 100, hole=0.45, showlegend=True, sort=False, marker=dict(colors=color, line=dict(color=pal, width=2.5)), hovertemplate='%{label} Accounts: %{value:.2f}%<extra></extra>'))
fig.update_layout(template=temp, title='Transport state Distribution', legend=dict(traceorder='reversed', y=1.05, x=0), uniformtext_minsize=15, uniformtext_mode='hide', width=700)
fig.show()
target = train.VIP.value_counts(normalize=True)
target.rename(index={1: 'True', 0: 'False'}, inplace=True)
(pal, color) = (['aliceblue', 'cornsilk', 'honeydew', 'plum'], ['skyblue', 'gold', 'darkseagreen', 'blueviolet'])
fig = go.Figure()
fig.add_trace(go.Pie(labels=target.index, values=target * 100, hole=0.45, showlegend=True, sort=False, marker=dict(colors=color, line=dict(color=pal, width=2.5)), hovertemplate='%{label} Accounts: %{value:.2f}%<extra></extra>'))
fig.update_layout(template=temp, title='VIP Distribution', legend=dict(traceorder='reversed', y=1.05, x=0), uniformtext_minsize=15, uniformtext_mode='hide', width=700)
fig.show()
target = train.loc[:, ['VIP', 'Transported']]
target['Transported'] = target['Transported'].astype(object)
target['VIP'] = target['VIP'].astype(object)
target = pd.get_dummies(target, columns=['Transported'])
target = target.groupby('VIP', as_index=False).agg('sum')
(pal, color) = (['aliceblue', 'mistyrose', 'cornsilk', 'honeydew', 'plum'], ['skyblue', 'salmon', 'gold', 'darkseagreen', 'blueviolet'])
rgb = ['rgba' + str(matplotlib.colors.to_rgba(i, 0.7)) for i in pal]
fig = go.Figure()
fig.add_trace(go.Bar(x=target.VIP, y=target.Transported_1, name='True', text=target.Transported_1, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[0], line=dict(color=pal[0], width=1.5)), hovertemplate='<b>%{x}</b><br>True accounts: %{y:.2f}'))
fig.add_trace(go.Bar(x=target.VIP, y=target.Transported_0, name='False', text=target.Transported_0, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[2], line=dict(color=pal[2], width=1.5)), hovertemplate='<b>%{x}</b><br>False accounts: %{y:.2f}'))
fig.update_layout(template=temp, title='Distribution of Transported', barmode='relative', width=1400, legend=dict(orientation='h', traceorder='reversed', yanchor='bottom', y=1.1, xanchor='left', x=0))
fig.show()
target = train.CryoSleep.value_counts(normalize=True)
target.rename(index={1: 'True', 0: 'False'}, inplace=True)
(pal, color) = (['seashell', 'aliceblue', 'mistyrose', 'cornsilk', 'honeydew', 'plum'], ['sandybrown', 'skyblue', 'salmon', 'gold', 'darkseagreen', 'blueviolet'])
fig = go.Figure()
fig.add_trace(go.Pie(labels=target.index, values=target * 100, hole=0.45, showlegend=True, sort=False, marker=dict(colors=color, line=dict(color=pal, width=2.5)), hovertemplate='%{label} Accounts: %{value:.2f}%<extra></extra>'))
fig.update_layout(template=temp, title='CryoSleep Distribution', legend=dict(traceorder='reversed', y=1.05, x=0), uniformtext_minsize=15, uniformtext_mode='hide', width=700)
fig.show()
target = train.loc[:, ['CryoSleep', 'Transported']]
target['Transported'] = target['Transported'].astype(object)
target['CryoSleep'] = target['CryoSleep'].astype(object)
target = pd.get_dummies(target, columns=['Transported'])
target = target.groupby('CryoSleep', as_index=False).agg('sum')
rgb = ['rgba' + str(matplotlib.colors.to_rgba(i, 0.7)) for i in pal]
fig = go.Figure()
fig.add_trace(go.Bar(x=target.CryoSleep, y=target.Transported_1, name='True', text=target.Transported_1, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[4], line=dict(color=pal[4], width=1.5)), hovertemplate='<b>%{x}</b><br>True accounts: %{y:.2f}'))
fig.add_trace(go.Bar(x=target.CryoSleep, y=target.Transported_0, name='False', text=target.Transported_0, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[3], line=dict(color=pal[3], width=1.5)), hovertemplate='<b>%{x}</b><br>False accounts: %{y:.2f}'))
fig.update_layout(template=temp, title='Distribution of Transported', barmode='relative', width=1400, legend=dict(orientation='h', traceorder='reversed', yanchor='bottom', y=1.1, xanchor='left', x=0))
fig.show()
target = train.HomePlanet.value_counts(normalize=True)
target.rename(index={1: 'True', 0: 'False'}, inplace=True)
(pal, color) = (['aliceblue', 'mistyrose', 'cornsilk', 'honeydew', 'plum'], ['skyblue', 'salmon', 'gold', 'darkseagreen', 'blueviolet'])
fig = go.Figure()
fig.add_trace(go.Pie(labels=target.index, values=target * 100, hole=0.45, showlegend=True, sort=False, marker=dict(colors=color, line=dict(color=pal, width=2.5)), hovertemplate='%{label} Accounts: %{value:.2f}%<extra></extra>'))
fig.update_layout(template=temp, title='HomePlanet Distribution', legend=dict(traceorder='reversed', y=1.05, x=0), uniformtext_minsize=15, uniformtext_mode='hide', width=700)
fig.show()
target = train.loc[:, ['HomePlanet', 'Transported']]
target['Transported'] = target['Transported'].astype(object)
target = pd.get_dummies(target, columns=['Transported'])
target = target.groupby('HomePlanet', as_index=False).agg('sum')
(pal, color) = (['seashell', 'aliceblue', 'mistyrose', 'cornsilk', 'honeydew', 'plum'], ['sandybrown', 'skyblue', 'salmon', 'gold', 'darkseagreen', 'blueviolet'])
rgb = ['rgba' + str(matplotlib.colors.to_rgba(i, 0.7)) for i in pal]
fig = go.Figure()
fig.add_trace(go.Bar(x=target.HomePlanet, y=target.Transported_1, name='True', text=target.Transported_1, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[5], line=dict(color=pal[5], width=1.5)), hovertemplate='<b>%{x}</b><br>True accounts: %{y:.2f}'))
fig.add_trace(go.Bar(x=target.HomePlanet, y=target.Transported_0, name='False', text=target.Transported_0, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[3], line=dict(color=pal[3], width=1.5)), hovertemplate='<b>%{x}</b><br>False accounts: %{y:.2f}'))
fig.update_layout(template=temp, title='Distribution of Transported', barmode='relative', width=1400, legend=dict(orientation='h', traceorder='reversed', yanchor='bottom', y=1.1, xanchor='left', x=0))
fig.show()
target = train.Destination.value_counts(normalize=True)
target.rename(index={1: 'True', 0: 'False'}, inplace=True)
(pal, color) = (['aliceblue', 'cornsilk', 'honeydew', 'plum'], ['skyblue', 'gold', 'darkseagreen', 'blueviolet'])
fig = go.Figure()
fig.add_trace(go.Pie(labels=target.index, values=target * 100, hole=0.45, showlegend=True, sort=False, marker=dict(colors=color, line=dict(color=pal, width=2.5)), hovertemplate='%{label} Accounts: %{value:.2f}%<extra></extra>'))
fig.update_layout(template=temp, title='Destination Distribution', legend=dict(traceorder='reversed', y=1.05, x=0), uniformtext_minsize=15, uniformtext_mode='hide', width=700)
fig.show()
target = train.loc[:, ['Destination', 'Transported']]
target['Transported'] = target['Transported'].astype(object)
target = pd.get_dummies(target, columns=['Transported'])
target = target.groupby('Destination', as_index=False).agg('sum')
(pal, color) = (['seashell', 'aliceblue', 'mistyrose', 'cornsilk', 'honeydew', 'plum'], ['sandybrown', 'skyblue', 'salmon', 'gold', 'darkseagreen', 'blueviolet'])
rgb = ['rgba' + str(matplotlib.colors.to_rgba(i, 0.7)) for i in pal]
fig = go.Figure()
fig.add_trace(go.Bar(x=target.Destination, y=target.Transported_1, name='True', text=target.Transported_1, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[2], line=dict(color=pal[2], width=1.5)), hovertemplate='<b>%{x}</b><br>True accounts: %{y:.2f}'))
fig.add_trace(go.Bar(x=target.Destination, y=target.Transported_0, name='False', text=target.Transported_0, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[0], line=dict(color=pal[0], width=1.5)), hovertemplate='<b>%{x}</b><br>False accounts: %{y:.2f}'))
fig.update_layout(template=temp, title='Distribution of Transported', barmode='relative', width=1400, legend=dict(orientation='h', traceorder='reversed', yanchor='bottom', y=1.1, xanchor='left', x=0))
fig.show()
train = pd.get_dummies(train, columns=['HomePlanet', 'Destination'])
test = pd.get_dummies(test, columns=['HomePlanet', 'Destination'])
target = train.loc[:, ['Cabin_desk', 'Transported']]
target['Transported'] = target['Transported'].astype(object)
target = pd.get_dummies(target, columns=['Transported'])
target = target.groupby('Cabin_desk', as_index=False).agg('sum')
(pal, color) = (['seashell', 'aliceblue', 'mistyrose', 'cornsilk', 'honeydew', 'plum'], ['sandybrown', 'skyblue', 'salmon', 'gold', 'darkseagreen', 'blueviolet'])
rgb = ['rgba' + str(matplotlib.colors.to_rgba(i, 0.7)) for i in pal]
fig = go.Figure()
fig.add_trace(go.Bar(x=target.Cabin_desk, y=target.Transported_1, name='True', text=target.Transported_1, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[5], line=dict(color=pal[5], width=1.5)), hovertemplate='<b>%{x}</b><br>True accounts: %{y:.2f}'))
fig.add_trace(go.Bar(x=target.Cabin_desk, y=target.Transported_0, name='False', text=target.Transported_0, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[3], line=dict(color=pal[3], width=1.5)), hovertemplate='<b>%{x}</b><br>False accounts: %{y:.2f}'))
fig.update_layout(template=temp, title='Distribution of Transported in different Cabin_desk', barmode='relative', width=1400, legend=dict(orientation='h', traceorder='reversed', yanchor='bottom', y=1.1, xanchor='left', x=0))
fig.show()
target = train.loc[:, ['Cabin_side', 'Transported']]
target['Transported'] = target['Transported'].astype(object)
target['Cabin_side'] = target['Cabin_side'].astype(object)
target = pd.get_dummies(target, columns=['Transported'])
target = target.groupby('Cabin_side', as_index=False).agg('sum')
rgb = ['rgba' + str(matplotlib.colors.to_rgba(i, 0.7)) for i in pal]
fig = go.Figure()
fig.add_trace(go.Bar(x=target.Cabin_side, y=target.Transported_1, name='True', text=target.Transported_1, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[4], line=dict(color=pal[4], width=1.5)), hovertemplate='<b>%{x}</b><br>True accounts: %{y:.2f}'))
fig.add_trace(go.Bar(x=target.Cabin_side, y=target.Transported_0, name='False', text=target.Transported_0, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=color[3], line=dict(color=pal[3], width=1.5)), hovertemplate='<b>%{x}</b><br>False accounts: %{y:.2f}'))
fig.update_layout(template=temp, title='Distribution of Transported in different Cabin_side', barmode='relative', width=1400, legend=dict(orientation='h', traceorder='reversed', yanchor='bottom', y=1.1, xanchor='left', x=0))
fig.show()
target = train.loc[:, ['Cabin_zone', 'Transported']]
target['Transported'] = target['Transported'].astype(object)
target = pd.get_dummies(target, columns=['Transported'])
rgb = ['rgba' + str(matplotlib.colors.to_rgba(i, 0.7)) for i in pal]
fig = go.Figure()
fig.add_trace(go.Bar(x=target.Cabin_zone, y=target.Transported_1, name='True', text=target.Transported_1, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=pal[5], line=dict(color=color[5], width=1.5)), hovertemplate='<b>%{x}</b><br>True accounts: %{y:.2f}'))
fig.add_trace(go.Bar(x=target.Cabin_zone, y=target.Transported_0, name='False', text=target.Transported_0, texttemplate='%{text:.0f}', textposition='inside', insidetextanchor='middle', marker=dict(color=pal[3], line=dict(color=color[3], width=1.5)), hovertemplate='<b>%{x}</b><br>False accounts: %{y:.2f}'))
fig.update_layout(template=temp, title='Distribution of Transported in different Cabin_zone', barmode='relative', width=1400, legend=dict(orientation='h', traceorder='reversed', yanchor='bottom', y=1.1, xanchor='left', x=0))
fig.show()
Room = train.loc[:, ['RoomService', 'Transported']]
Food = train.loc[:, ['FoodCourt', 'Transported']]
shop = train.loc[:, ['ShoppingMall', 'Transported']]
spa = train.loc[:, ['Spa', 'Transported']]
vr = train.loc[:, ['VRDeck', 'Transported']]
Room['consume_state'] = 0
Room.loc[Room['RoomService'] > 0, ['consume_state']] = 1
Food['consume_state'] = 0
Food.loc[Food['FoodCourt'] > 0, ['consume_state']] = 1
shop['consume_state'] = 0
shop.loc[shop['ShoppingMall'] > 0, ['consume_state']] = 1
spa['consume_state'] = 0
spa.loc[spa['Spa'] > 0, ['consume_state']] = 1
vr['consume_state'] = 0
vr.loc[vr['VRDeck'] > 0, ['consume_state']] = 1
plt.style.use('ggplot')
(f, (ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1, 5, figsize=(20, 10), dpi=100)
a = Room['consume_state'].sum()
b = len(Room['consume_state']) - a
ax1.pie([a, b], labels=['consume', 'not consume'], colors=['skyblue', 'salmon'], autopct='%.1f%%')
ax1.set_title('RoomService')
a = Food['consume_state'].sum()
b = len(Food['consume_state']) - a
ax2.pie([a, b], labels=['consume', 'not consume'], colors=['skyblue', 'salmon'], autopct='%.1f%%')
ax2.set_title('FoodCourt')
a = shop['consume_state'].sum()
b = len(shop['consume_state']) - a
ax3.pie([a, b], labels=['consume', 'not consume'], colors=['skyblue', 'salmon'], autopct='%.1f%%')
ax3.set_title('ShoppingMall')
a = spa['consume_state'].sum()
b = len(spa['consume_state']) - a
ax4.pie([a, b], labels=['consume', 'not consume'], colors=['skyblue', 'salmon'], autopct='%.1f%%')
ax4.set_title('SPA')
a = vr['consume_state'].sum()
b = len(vr['consume_state']) - a
ax5.pie([a, b], labels=['consume', 'not consume'], colors=['skyblue', 'salmon'], autopct='%.1f%%')
ax5.set_title('VR')
df = pd.concat([Room, Food, shop, spa, vr], axis=1)
df['sum'] = df.iloc[:, 2] + df.iloc[:, 5] + df.iloc[:, 8]
df = df.iloc[:, [0, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 15]]
plt.style.use('ggplot')
(f, (ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(1, 6, figsize=(30, 20), dpi=100)
a = df.loc[df['sum'] == 0, 'Transported'].sum()
b = 8693 - a
ax1.pie([a, b], labels=['True', 'False'], colors=['skyblue', 'salmon'], autopct='%.1f%%')
ax1.set_title('0 amenity')
a = df.loc[df['sum'] == 1, 'Transported'].sum()
b = 8693 - a
ax2.pie([a, b], labels=['True', 'False'], colors=['skyblue', 'salmon'], autopct='%.1f%%')
ax2.set_title('1 amenity')
a = df.loc[df['sum'] == 2, 'Transported'].sum()
b = 8693 - a
ax3.pie([a, b], labels=['True', 'False'], colors=['skyblue', 'salmon'], autopct='%.1f%%')
ax3.set_title('2 amenity')
a = df.loc[df['sum'] == 3, 'Transported'].sum()
b = 8693 - a
ax4.pie([a, b], labels=['True', 'False'], colors=['skyblue', 'salmon'], autopct='%.1f%%')
ax4.set_title('3 amenity')
a = df.loc[df['sum'] == 4, 'Transported'].sum()
b = 8693 - a
ax5.pie([a, b], labels=['True', 'False'], colors=['skyblue', 'salmon'], autopct='%.1f%%')
ax5.set_title('4 amenity')
a = df.loc[df['sum'] == 5, 'Transported'].sum()
b = 8693 - a
ax6.pie([a, b], labels=['True', 'False'], colors=['skyblue', 'salmon'], autopct='%.1f%%')
ax6.set_title('5 amenity')
train = pd.concat([train, df['sum']], axis=1)
train
Room1 = Room.loc[Room['consume_state'] > 0, :]
Food1 = Food.loc[Food['consume_state'] > 0, :]
shop1 = shop.loc[shop['consume_state'] > 0, :]
spa1 = spa.loc[spa['consume_state'] > 0, :]
vr1 = vr.loc[vr['consume_state'] > 0, :]
Room1 = Room1.describe().T
Food1 = Food1.describe().T
shop1 = shop1.describe().T
spa1 = spa1.describe().T
vr1 = vr1.describe().T
df = pd.concat([Room1, Food1, shop1, spa1, vr1])
df
cor = train.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(cor, cmap='coolwarm', vmin=-1, vmax=1, center=0, annot=True, fmt='.2f', annot_kws={'fontsize': 10})