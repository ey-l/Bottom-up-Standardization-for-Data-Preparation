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
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
train_df.head(1)

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
data_glimpse(train_df)

def parsing_from(dataset, idx):
    return dataset['Cabin'].str.split('/').str[idx]
train_df['Deck'] = parsing_from(train_df, 0)
train_df['DeckNumber'] = parsing_from(train_df, 1)
train_df['Side'] = parsing_from(train_df, 2)
train_df.head(1)
train_df.drop(['Cabin'], axis=1, inplace=True)
train_df.columns
test_df['Deck'] = parsing_from(test_df, 0)
test_df['DeckNumber'] = parsing_from(test_df, 1)
test_df['Side'] = parsing_from(test_df, 2)
test_df.drop(['Cabin'], axis=1, inplace=True)
train_df.drop(['Name'], axis=1, inplace=True)
test_df.drop(['Name'], axis=1, inplace=True)

def fig_layout(title, xaxis, yaxis):
    fig.update_layout({'title': {'text': title, 'x': 0.5, 'y': 0.9, 'font': {'size': 15}}, 'xaxis': {'title': xaxis, 'showticklabels': True, 'tickfont': {'size': 9}}, 'yaxis': {'title': yaxis, 'tickfont': {'size': 10}}, 'template': 'plotly_dark'})
missing_val = train_df.isnull().sum().sort_values(ascending=False)
fig = go.Figure()
fig.add_trace(go.Bar(x=missing_val.index, y=missing_val, text=missing_val))
title = '<b>Count of missing values by features</b>'
xaxis = 'Variables'
yaxis = 'Count of missing values'
fig_layout(title, xaxis, yaxis)
fig.show()
y = train_df['Transported']
train_df = train_df.drop(columns=['Transported'], axis=1)
print(f'Shape of train dataset : {train_df.shape}')
print(f'Shape of test dataset : {test_df.shape}')
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
test_df = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns, index=test_df.index)
train_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'DeckNumber']] = train_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'DeckNumber']].apply(pd.to_numeric)
test_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'DeckNumber']] = test_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'DeckNumber']].apply(pd.to_numeric)
print(f'Missing values of train_df : {train_df.isnull().sum().sum()}')
print(f'Missing values of test_df : {test_df.isnull().sum().sum()}')
train_df = pd.concat([train_df, y], axis=1)
train_df.head(1)
fig = px.histogram(train_df, 'Transported', color='Transported')
title = 'Histogram of Target feature'
xaxis = 'Transported'
yaxis = 'Count'
fig_layout(title, xaxis, yaxis)
fig.show()
continuous_fea = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_fea = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
train_df[continuous_fea].describe()
import plotly.figure_factory as ff
fig = ff.create_distplot([train_df['Age']], ['Age'], bin_size=5, curve_type='normal')
title = '<b>Distplot of Age</b>'
xaxis = 'Age'
yaxis = '%'
fig_layout(title, xaxis, yaxis)
fig.show()
fig = px.histogram(train_df, x='Age', color='Transported')
title = '<b>Distplot of Age by target</b>'
xaxis = 'Age'
yaxis = '%'
fig_layout(title, xaxis, yaxis)
fig.show()
fig = ff.create_distplot([np.sqrt(train_df['RoomService'])], ['RoomService'], bin_size=10, curve_type='normal')
title = '<b>Distplot of RoomService</b>'
xaxis = 'RoomService'
yaxis = '%'
fig_layout(title, xaxis, yaxis)
fig.show()
from scipy.stats import skew
for fea in continuous_fea:
    print(f'Skewness of {fea} is {skew(np.array(train_df[fea]))}')

def impute_outlier(col):
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    return Q3 + 1.5 * IQR
outlier_fea = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for fea in outlier_fea:
    max_val = impute_outlier(fea)
    train_df[fea] = train_df[fea].map(lambda x: max_val if x > max_val else x)
    test_df[fea] = test_df[fea].map(lambda x: max_val if x > max_val else x)
fig = ff.create_distplot([np.sqrt(train_df['RoomService'])], ['RoomService'], bin_size=3, curve_type='normal')
title = '<b>Distplot of RoomService</b>'
xaxis = 'RoomService'
yaxis = '%'
fig_layout(title, xaxis, yaxis)
fig.show()
from plotly.subplots import make_subplots

def create_count_plot(fea):
    grouped_df = train_df.groupby(fea).size().reset_index()
    grouped_df.columns = [fea, 'Count']
    grouped_df_target = train_df.groupby([fea, 'Transported']).size().reset_index()
    grouped_df_target.columns = [fea, 'Transported', 'Count']
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Bar(x=grouped_df[fea], y=grouped_df['Count'], name=fea), row=1, col=1)
    for trans in train_df['Transported'].unique():
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
y_train = np.where(train_df['Transported'] == True, 1, 0)
X = train_df.drop(columns=['Transported'], axis=1)
X_test = test_df
print(f'Size of each table : y = {y.shape}, X = {X.shape}, X_test = {X_test.shape}')
train_df['Transported'] = np.where(train_df['Transported'] == True, 1, 0)
corr_fea = continuous_fea
corr_fea.append('Transported')
corr = train_df[corr_fea].corr()
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