import numpy as np
import pandas as pd
import plotly.express as px
titanic = pd.read_csv('data/input/spaceship-titanic/train.csv')
titanic_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
titanic.head(10)
titanic.info()
titanic.nunique().plot.bar(title='Cardinality in our columns')
print(f'Duplicates in train set: {titanic.duplicated().sum()}, ({np.round(100 * titanic.duplicated().sum() / len(titanic), 1)}%)')
print('')
print(f'Duplicates in test set: {titanic_test.duplicated().sum()}, ({np.round(100 * titanic_test.duplicated().sum() / len(titanic_test), 1)}%)')
titanic[['Passenger_group', 'Passennger_no._in_group']] = titanic['PassengerId'].str.split('_', expand=True)
titanic.drop(columns=['PassengerId'], inplace=True)
titanic.head()
titanic[['Cabin_deck', 'Cabin_no.', 'Cabin_side']] = titanic['Cabin'].str.split('/', expand=True)
titanic.drop(columns=['Cabin'], inplace=True)
titanic.head()
titanic[['firstname', 'last_name']] = titanic['Name'].str.split(' ', expand=True)
titanic.drop(columns=['Name'], inplace=True)
titanic.head()
titanic.nunique()
titanic.nunique().plot.bar(title='Cardinality in our columns')
missing = (titanic.isnull().mean().sort_values(ascending=False) * 100).reset_index()
missing.rename(columns={0: 'Average'}, inplace=True)
missing.head()
fig = px.histogram(missing, x='Average', y='index', title='<b>% of Missing values', color='index', labels={'Average': '%age of missing values', 'index': 'Column Names'})
fig.update_layout(font_color='white', font_size=12, title_font_color='cyan', legend_title_font_color='white', legend_title_font_size=20, template='plotly_dark', title_font_size=30)
fig.update_layout(xaxis_title='<b>Amount in %age', xaxis_title_font_size=20, yaxis_title='<b>Column-Name', yaxis_title_font_size=20, title_x=0.5)
fig.show()
fig = px.imshow(titanic.isnull().T, color_continuous_scale=px.colors.sequential.Electric, title='<b>Missing values in our data')
fig.update_layout(template='plotly_dark', title_font_size=30, title_x=0.5)
fig.show()
titanic.dropna().shape[0] / titanic.shape[0] * 100
titanic['Transported'] = np.where(titanic['Transported'] == True, 1, 0)
titanic.info()
titanic.head()
numerical_cols = ['Age', 'RoomService', 'Spa', 'VRDeck', 'ShoppingMall', 'FoodCourt']
numerical_cols
categorical_cols = [feature for feature in titanic.columns if feature not in numerical_cols]
categorical_cols
for feature in categorical_cols:
    if feature not in ['Passenger_group', 'firstname', 'last_name', 'Cabin_no.']:
        df = titanic[feature].value_counts().reset_index()
        fig = px.pie(df, values=df.columns[1], names=df.columns[0], labels={'index': '<b>' + feature, feature: 'Count'}, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(title='<b>' + feature, title_font_size=30, font_size=20, title_x=0.5, legend_bordercolor='#000', legend_borderwidth=2, hoverlabel_font_size=20)
        fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
        fig.show()
for feature in categorical_cols:
    if feature not in ['Passenger_group', 'firstname', 'last_name', 'Cabin_no.', 'Transported']:
        fig = px.histogram(titanic, x=feature, facet_col='Transported', color='Transported', color_discrete_sequence=px.colors.qualitative.Alphabet_r)
        fig.update_layout(title='<b>' + feature + ' vs Transported', title_font_size=30, font_size=20, title_x=0.5, hoverlabel_font_size=20, template='plotly_dark')
        fig.show()
for feature in numerical_cols:
    fig = px.violin(titanic, x=feature, color='Transported', title='<b>' + feature + ' Distribution', template='plotly_dark')
    fig.update_layout(hovermode='x', title_font_size=30)
    fig.update_layout(title_font_color='#ffff00', template='plotly_dark', title_font_size=30, hoverlabel_font_size=20, title_x=0.5)
    fig.show()
    fig = px.histogram(titanic, x=feature, title='<b>' + feature + 'Vs Transported', color='Transported', template='plotly_dark')
    fig.update_layout(hovermode='x', title_font_size=30)
    fig.update_layout(title_font_color='#ffff00', template='plotly_dark', title_font_size=30, hoverlabel_font_size=20, title_x=0.5)
    fig.show()
px.imshow(titanic.corr().round(3), text_auto=True)
px.scatter_matrix(titanic[numerical_cols + ['Transported']], height=800, color='Transported')
fig = px.parallel_coordinates(titanic, color='Transported', title='<b>Multivariate plot for Numerical Data')
fig.update_layout(title_font_size=30, title_x=0.5)
fig = px.imshow(pd.crosstab(titanic['Cabin_deck'], titanic['Cabin_side']).T, text_auto=True, title='No. of people Travelling in (Cabin_DecK,Cabin_side)', labels={'color': '<b>No.of people'}, color_continuous_scale=px.colors.sequential.haline_r)
fig.update_layout(font_size=15, font_color='#ffcce6', title_font_size=30, title_font_color='Orange', template='plotly_dark')
fig.show()
fig = px.imshow(pd.crosstab(titanic['Cabin_deck'], titanic['Cabin_side'], titanic['Transported'], aggfunc='mean').round(3).T, text_auto=True, title='% of people Transported(1) in (Cabin_DecK,Cabin_side)', labels={'color': '<b>%Transpoted'}, color_continuous_scale=px.colors.sequential.haline_r)
fig.update_layout(font_size=15, font_color='#ffcce6', title_font_size=30, title_font_color='Orange', template='plotly_dark')
fig.show()
fig = px.imshow(pd.crosstab(titanic['HomePlanet'], titanic['CryoSleep']).T, text_auto=True, title='<b>No. of people Travelling in (HomePlanet,Cryosleep)', labels={'color': '<b>No.of people'}, color_continuous_scale=px.colors.sequential.haline_r)
fig.update_layout(font_size=15, font_color='#ffcce6', title_font_size=30, title_font_color='Orange', template='plotly_dark')
fig.show()
fig = px.imshow(pd.crosstab(titanic['HomePlanet'], titanic['CryoSleep'], titanic['Transported'], aggfunc='mean').round(3).T, text_auto=True, title='<b>No. of people Travelling in (HomePlanet,Cryosleep)', labels={'color': '<b>%Transpoted'}, color_continuous_scale=px.colors.sequential.haline_r)
fig.update_layout(font_size=15, font_color='#ffcce6', title_font_size=30, title_font_color='Orange', template='plotly_dark')
fig.show()
titanic[(titanic['HomePlanet'] == 'Europa') & (titanic['CryoSleep'] == False)]['Transported'].mean()
fig = px.parallel_categories(titanic, color='Transported', title='<b>Multivarite Plot For Categorical data', height=600, width=1000)
fig.update_layout(title_font_size=30, title_x=0.5)