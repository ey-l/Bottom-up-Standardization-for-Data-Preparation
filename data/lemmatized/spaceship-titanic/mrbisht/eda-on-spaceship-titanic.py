import numpy as np
import pandas as pd
import plotly.express as px
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head(10)
_input1.info()
_input1.nunique().plot.bar(title='Cardinality in our columns')
print(f'Duplicates in train set: {_input1.duplicated().sum()}, ({np.round(100 * _input1.duplicated().sum() / len(_input1), 1)}%)')
print('')
print(f'Duplicates in test set: {_input0.duplicated().sum()}, ({np.round(100 * _input0.duplicated().sum() / len(_input0), 1)}%)')
_input1[['Passenger_group', 'Passennger_no._in_group']] = _input1['PassengerId'].str.split('_', expand=True)
_input1 = _input1.drop(columns=['PassengerId'], inplace=False)
_input1.head()
_input1[['Cabin_deck', 'Cabin_no.', 'Cabin_side']] = _input1['Cabin'].str.split('/', expand=True)
_input1 = _input1.drop(columns=['Cabin'], inplace=False)
_input1.head()
_input1[['firstname', 'last_name']] = _input1['Name'].str.split(' ', expand=True)
_input1 = _input1.drop(columns=['Name'], inplace=False)
_input1.head()
_input1.nunique()
_input1.nunique().plot.bar(title='Cardinality in our columns')
missing = (_input1.isnull().mean().sort_values(ascending=False) * 100).reset_index()
missing = missing.rename(columns={0: 'Average'}, inplace=False)
missing.head()
fig = px.histogram(missing, x='Average', y='index', title='<b>% of Missing values', color='index', labels={'Average': '%age of missing values', 'index': 'Column Names'})
fig.update_layout(font_color='white', font_size=12, title_font_color='cyan', legend_title_font_color='white', legend_title_font_size=20, template='plotly_dark', title_font_size=30)
fig.update_layout(xaxis_title='<b>Amount in %age', xaxis_title_font_size=20, yaxis_title='<b>Column-Name', yaxis_title_font_size=20, title_x=0.5)
fig.show()
fig = px.imshow(_input1.isnull().T, color_continuous_scale=px.colors.sequential.Electric, title='<b>Missing values in our data')
fig.update_layout(template='plotly_dark', title_font_size=30, title_x=0.5)
fig.show()
_input1.dropna().shape[0] / _input1.shape[0] * 100
_input1['Transported'] = np.where(_input1['Transported'] == True, 1, 0)
_input1.info()
_input1.head()
numerical_cols = ['Age', 'RoomService', 'Spa', 'VRDeck', 'ShoppingMall', 'FoodCourt']
numerical_cols
categorical_cols = [feature for feature in _input1.columns if feature not in numerical_cols]
categorical_cols
for feature in categorical_cols:
    if feature not in ['Passenger_group', 'firstname', 'last_name', 'Cabin_no.']:
        df = _input1[feature].value_counts().reset_index()
        fig = px.pie(df, values=df.columns[1], names=df.columns[0], labels={'index': '<b>' + feature, feature: 'Count'}, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(title='<b>' + feature, title_font_size=30, font_size=20, title_x=0.5, legend_bordercolor='#000', legend_borderwidth=2, hoverlabel_font_size=20)
        fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
        fig.show()
for feature in categorical_cols:
    if feature not in ['Passenger_group', 'firstname', 'last_name', 'Cabin_no.', 'Transported']:
        fig = px.histogram(_input1, x=feature, facet_col='Transported', color='Transported', color_discrete_sequence=px.colors.qualitative.Alphabet_r)
        fig.update_layout(title='<b>' + feature + ' vs Transported', title_font_size=30, font_size=20, title_x=0.5, hoverlabel_font_size=20, template='plotly_dark')
        fig.show()
for feature in numerical_cols:
    fig = px.violin(_input1, x=feature, color='Transported', title='<b>' + feature + ' Distribution', template='plotly_dark')
    fig.update_layout(hovermode='x', title_font_size=30)
    fig.update_layout(title_font_color='#ffff00', template='plotly_dark', title_font_size=30, hoverlabel_font_size=20, title_x=0.5)
    fig.show()
    fig = px.histogram(_input1, x=feature, title='<b>' + feature + 'Vs Transported', color='Transported', template='plotly_dark')
    fig.update_layout(hovermode='x', title_font_size=30)
    fig.update_layout(title_font_color='#ffff00', template='plotly_dark', title_font_size=30, hoverlabel_font_size=20, title_x=0.5)
    fig.show()
px.imshow(_input1.corr().round(3), text_auto=True)
px.scatter_matrix(_input1[numerical_cols + ['Transported']], height=800, color='Transported')
fig = px.parallel_coordinates(_input1, color='Transported', title='<b>Multivariate plot for Numerical Data')
fig.update_layout(title_font_size=30, title_x=0.5)
fig = px.imshow(pd.crosstab(_input1['Cabin_deck'], _input1['Cabin_side']).T, text_auto=True, title='No. of people Travelling in (Cabin_DecK,Cabin_side)', labels={'color': '<b>No.of people'}, color_continuous_scale=px.colors.sequential.haline_r)
fig.update_layout(font_size=15, font_color='#ffcce6', title_font_size=30, title_font_color='Orange', template='plotly_dark')
fig.show()
fig = px.imshow(pd.crosstab(_input1['Cabin_deck'], _input1['Cabin_side'], _input1['Transported'], aggfunc='mean').round(3).T, text_auto=True, title='% of people Transported(1) in (Cabin_DecK,Cabin_side)', labels={'color': '<b>%Transpoted'}, color_continuous_scale=px.colors.sequential.haline_r)
fig.update_layout(font_size=15, font_color='#ffcce6', title_font_size=30, title_font_color='Orange', template='plotly_dark')
fig.show()
fig = px.imshow(pd.crosstab(_input1['HomePlanet'], _input1['CryoSleep']).T, text_auto=True, title='<b>No. of people Travelling in (HomePlanet,Cryosleep)', labels={'color': '<b>No.of people'}, color_continuous_scale=px.colors.sequential.haline_r)
fig.update_layout(font_size=15, font_color='#ffcce6', title_font_size=30, title_font_color='Orange', template='plotly_dark')
fig.show()
fig = px.imshow(pd.crosstab(_input1['HomePlanet'], _input1['CryoSleep'], _input1['Transported'], aggfunc='mean').round(3).T, text_auto=True, title='<b>No. of people Travelling in (HomePlanet,Cryosleep)', labels={'color': '<b>%Transpoted'}, color_continuous_scale=px.colors.sequential.haline_r)
fig.update_layout(font_size=15, font_color='#ffcce6', title_font_size=30, title_font_color='Orange', template='plotly_dark')
fig.show()
_input1[(_input1['HomePlanet'] == 'Europa') & (_input1['CryoSleep'] == False)]['Transported'].mean()
fig = px.parallel_categories(_input1, color='Transported', title='<b>Multivarite Plot For Categorical data', height=600, width=1000)
fig.update_layout(title_font_size=30, title_x=0.5)