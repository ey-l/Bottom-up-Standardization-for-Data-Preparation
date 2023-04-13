import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
pd.options.display.max_columns = 200
import cufflinks as cf
cf.go_offline(connected=True)
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
pyo.init_notebook_mode()
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input1.head(1)

def missing(df):
    """
    This function shows number of missing values and its percetages 
    """
    missing_number = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_number', 'Missing_percent'])
    return missing_values

def categorize(df):
    """
    This function shows number of features by dtypes.
    Result of function is not always accruate because this result estimate dtypes before preprocessing.
    """
    Quantitive_features = df.select_dtypes([np.number]).columns.tolist()
    Categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
    Discrete_features = [col for col in Quantitive_features if len(df[col].unique()) < 10]
    Continuous_features = [col for col in Quantitive_features if col not in Discrete_features]
    print(f'Quantitive feautres : {Quantitive_features} \nDiscrete features : {Discrete_features} \nContinous features : {Continuous_features} \nCategorical features : {Categorical_features}\n')
    print(f'Number of quantitive feautres : {len(Quantitive_features)} \nNumber of discrete features : {len(Discrete_features)} \nNumber of continous features : {len(Continuous_features)} \nNumber of categorical features : {len(Categorical_features)}')

def unique(df):
    """
    This function returns table storing number of unique values and its samples.
    """
    tb1 = pd.DataFrame({'Columns': df.columns, 'Number_of_Unique': df.nunique().values.tolist(), 'Sample1': df.sample(1).values.tolist()[0], 'Sample2': df.sample(1).values.tolist()[0], 'Sample3': df.sample(1).values.tolist()[0], 'Sample4': df.sample(1).values.tolist()[0], 'Sample5': df.sample(1).values.tolist()[0]})
    return tb1

def data_glimpse(df):
    print('1. Dataset Preview \n')
    print('-------------------------------------------------------------------------------\n')
    print('2. Column Information \n')
    print(f'Dataset have {df.shape[0]} rows and {df.shape[1]} columns')
    print('\n')
    print(f'Dataset Column name : {df.columns.values}')
    print('\n')
    categorize(df)
    print('-------------------------------------------------------------------------------\n')
    print('3. Missing data table : \n')
    print('-------------------------------------------------------------------------------\n')
    print('4. Number of unique value by column : \n')
    print('-------------------------------------------------------------------------------\n')
    print('5. Describe table : \n')
    print('-------------------------------------------------------------------------------\n')
    print(df.info())
    print('-------------------------------------------------------------------------------\n')
data_glimpse(_input1)

def parsing_from(dataset, idx):
    return dataset['Cabin'].str.split('/').str[idx]
_input1['Deck'] = parsing_from(_input1, 0)
_input1['DeckNumber'] = parsing_from(_input1, 1)
_input1['Side'] = parsing_from(_input1, 2)
_input1.head(1)
_input1 = _input1.drop(['Cabin'], axis=1, inplace=False)
_input1.columns
_input0['Deck'] = parsing_from(_input0, 0)
_input0['DeckNumber'] = parsing_from(_input0, 1)
_input0['Side'] = parsing_from(_input0, 2)
_input0 = _input0.drop(['Cabin'], axis=1, inplace=False)
_input1 = _input1.drop(['Name'], axis=1, inplace=False)
_input0 = _input0.drop(['Name'], axis=1, inplace=False)

def fig_layout(title, xaxis, yaxis):
    fig.update_layout({'title': {'text': title, 'x': 0.5, 'y': 0.9, 'font': {'size': 15}}, 'xaxis': {'title': xaxis, 'showticklabels': True, 'tickfont': {'size': 9}}, 'yaxis': {'title': yaxis, 'tickfont': {'size': 10}}, 'template': 'plotly_dark'})
missing_val = _input1.isnull().sum().sort_values(ascending=False)
fig = go.Figure()
fig.add_trace(go.Bar(x=missing_val.index, y=missing_val, text=missing_val))
title = '<b>Count of missing values by features</b>'
xaxis = 'Variables'
yaxis = 'Count of missing values'
fig_layout(title, xaxis, yaxis)
fig.show()
y = _input1['Transported']
_input1 = _input1.drop(columns=['Transported'], axis=1)
print(f'Shape of train dataset : {_input1.shape}')
print(f'Shape of test dataset : {_input0.shape}')
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
_input1 = pd.DataFrame(imputer.fit_transform(_input1), columns=_input1.columns, index=_input1.index)
_input0 = pd.DataFrame(imputer.transform(_input0), columns=_input0.columns, index=_input0.index)
_input1[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'DeckNumber']] = _input1[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'DeckNumber']].apply(pd.to_numeric)
_input0[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'DeckNumber']] = _input0[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'DeckNumber']].apply(pd.to_numeric)
print(f'Missing values of train_df : {_input1.isnull().sum().sum()}')
print(f'Missing values of test_df : {_input0.isnull().sum().sum()}')
_input1 = pd.concat([_input1, y], axis=1)
_input1.head(1)
fig = px.histogram(_input1, 'Transported', color='Transported')
title = 'Histogram of Target feature'
xaxis = 'Transported'
yaxis = 'Count'
fig_layout(title, xaxis, yaxis)
fig.show()
continuous_fea = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_fea = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
_input1[continuous_fea].describe()
import plotly.figure_factory as ff
fig = ff.create_distplot([_input1['Age']], ['Age'], bin_size=5, curve_type='normal')
title = '<b>Distplot of Age</b>'
xaxis = 'Age'
yaxis = '%'
fig_layout(title, xaxis, yaxis)
fig.show()
fig = px.histogram(_input1, x='Age', color='Transported')
title = '<b>Distplot of Age by target</b>'
xaxis = 'Age'
yaxis = '%'
fig_layout(title, xaxis, yaxis)
fig.show()
fig = ff.create_distplot([np.sqrt(_input1['RoomService'])], ['RoomService'], bin_size=10, curve_type='normal')
title = '<b>Distplot of RoomService</b>'
xaxis = 'RoomService'
yaxis = '%'
fig_layout(title, xaxis, yaxis)
fig.show()
from scipy.stats import skew
for fea in continuous_fea:
    print(f'Skewness of {fea} is {skew(np.array(_input1[fea]))}')

def impute_outlier(col):
    Q1 = _input1[col].quantile(0.25)
    Q3 = _input1[col].quantile(0.75)
    IQR = Q3 - Q1
    return Q3 + 1.5 * IQR
outlier_fea = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for fea in outlier_fea:
    max_val = impute_outlier(fea)
    _input1[fea] = _input1[fea].map(lambda x: max_val if x > max_val else x)
    _input0[fea] = _input0[fea].map(lambda x: max_val if x > max_val else x)
fig = ff.create_distplot([np.sqrt(_input1['RoomService'])], ['RoomService'], bin_size=3, curve_type='normal')
title = '<b>Distplot of RoomService</b>'
xaxis = 'RoomService'
yaxis = '%'
fig_layout(title, xaxis, yaxis)
fig.show()
from plotly.subplots import make_subplots

def create_count_plot(fea):
    grouped_df = _input1.groupby(fea).size().reset_index()
    grouped_df.columns = [fea, 'Count']
    grouped_df_target = _input1.groupby([fea, 'Transported']).size().reset_index()
    grouped_df_target.columns = [fea, 'Transported', 'Count']
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Bar(x=grouped_df[fea], y=grouped_df['Count'], name=fea), row=1, col=1)
    for trans in _input1['Transported'].unique():
        plot_df = grouped_df_target[grouped_df_target['Transported'] == trans]
        fig.add_trace(go.Bar(x=plot_df[fea], y=plot_df['Count'], name=f'Transported {trans}'), row=1, col=2)
    fig.update_layout({'title': {'text': f'Countplots of {fea}', 'x': 0.5, 'y': 0.9, 'font': {'size': 15}}, 'yaxis': {'title': 'Count', 'tickfont': {'size': 10}}, 'template': 'plotly_dark'})
    fig.update_xaxes(title_text=fea, row=1, col=1)
    fig.update_xaxes(title_text=fea, row=1, col=2)
    fig.show()
create_count_plot('HomePlanet')
create_count_plot('CryoSleep')
create_count_plot('Destination')
create_count_plot('VIP')
create_count_plot('Deck')
create_count_plot('Side')
y_train = np.where(_input1['Transported'] == True, 1, 0)
X = _input1.drop(columns=['Transported'], axis=1)
X_test = _input0
print(f'Size of each table : y = {y.shape}, X = {X.shape}, X_test = {X_test.shape}')
_input1['Transported'] = np.where(_input1['Transported'] == True, 1, 0)
corr_fea = continuous_fea
corr_fea.append('Transported')
corr = _input1[corr_fea].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
df_mask = corr.mask(mask)
fig = ff.create_annotated_heatmap(z=df_mask.round(3).to_numpy(), x=df_mask.columns.tolist(), y=df_mask.columns.tolist(), colorscale=px.colors.diverging.RdBu, hoverinfo='none', showscale=True, ygap=1, xgap=1)
fig.update_xaxes(side='bottom')
fig.update_layout(title_text='Heatmap', title_x=0.5, width=1000, height=1000, xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False, yaxis_autorange='reversed', template='plotly_dark')
for i in range(len(fig.layout.annotations)):
    if fig.layout.annotations[i].text == 'nan':
        fig.layout.annotations[i].text = ''
    fig.layout.annotations[i].font.size = 10
fig.show()
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)