import plotly.express as px
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()
import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x='sepal_width', y='sepal_length')
fig.show()
import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species', size='petal_length', hover_data=['petal_width'])
fig.show()
import plotly.graph_objects as go
import numpy as np
N = 1000
t = np.linspace(0, 10, 100)
y = np.sin(t)
fig = go.Figure(data=go.Scatter(x=t, y=y, mode='markers'))
fig.show()
import plotly.graph_objects as go
import numpy as np
np.random.seed(1)
N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N) - 5
fig = go.Figure()
fig.add_trace(go.Scatter(x=random_x, y=random_y0, mode='markers', name='markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y1, mode='lines+markers', name='lines+markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y2, mode='lines', name='lines'))
fig.show()
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode='markers', marker=dict(size=[40, 60, 80, 100], color=[0, 1, 2, 3])))
fig.show()
import plotly.graph_objects as go
import numpy as np
t = np.linspace(0, 10, 100)
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=np.sin(t), name='sin', mode='markers', marker_color='rgba(152, 0, 0, .8)'))
fig.add_trace(go.Scatter(x=t, y=np.cos(t), name='cos', marker_color='rgba(255, 182, 193, .9)'))
fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
fig.update_layout(title='Styled Scatter', yaxis_zeroline=False, xaxis_zeroline=False)
fig.show()
import plotly.graph_objects as go
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')
fig = go.Figure(data=go.Scatter(x=data['Postal'], y=data['Population'], mode='markers', marker_color=data['Population'], text=data['State']))
fig.update_layout(title='Population of USA States')
fig.show()
import plotly.graph_objects as go
import numpy as np
fig = go.Figure(data=go.Scatter(y=np.random.randn(500), mode='markers', marker=dict(size=16, color=np.random.randn(500), colorscale='Viridis', showscale=True)))
fig.show()
import plotly.graph_objects as go
import numpy as np
N = 100000
fig = go.Figure(data=go.Scattergl(x=np.random.randn(N), y=np.random.randn(N), mode='markers', marker=dict(color=np.random.randn(N), colorscale='Viridis', line_width=1)))
fig.show()
import plotly.graph_objects as go
import numpy as np
N = 100000
r = np.random.uniform(0, 1, N)
theta = np.random.uniform(0, 2 * np.pi, N)
fig = go.Figure(data=go.Scattergl(x=r * np.cos(theta), y=r * np.sin(theta), mode='markers', marker=dict(color=np.random.randn(N), colorscale='Viridis', line_width=1)))
fig.show()
import plotly.express as px
import numpy as np
t = np.linspace(0, 2 * np.pi, 100)
fig = px.line(x=t, y=np.cos(t), labels={'x': 't', 'y': 'cos(t)'})
fig.show()
import plotly.express as px
df = px.data.gapminder().query("continent == 'Oceania'")
fig = px.line(df, x='year', y='lifeExp', color='country')
fig.show()
import plotly.express as px
df = px.data.gapminder().query("continent != 'Asia'")
fig = px.line(df, x='year', y='lifeExp', color='continent', line_group='country', hover_name='country')
fig.show()
import plotly.express as px
df = px.data.stocks(indexed=True)
fig = px.line(df, facet_row='company', facet_row_spacing=0.01, height=200, width=200)
fig.update_xaxes(visible=False, fixedrange=True)
fig.update_yaxes(visible=False, fixedrange=True)
fig.update_layout(annotations=[], overwrite=True)
fig.update_layout(showlegend=False, plot_bgcolor='white', margin=dict(t=10, l=10, b=10, r=10))
fig.show(config=dict(displayModeBar=False))
import plotly.graph_objects as go
import numpy as np
x = np.arange(10)
fig = go.Figure(data=go.Scatter(x=x, y=x ** 2))
fig.show()
import plotly.graph_objects as go
month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
high_2000 = [32.5, 37.6, 49.9, 53.0, 69.1, 75.4, 76.5, 76.6, 70.7, 60.6, 45.1, 29.3]
low_2000 = [13.8, 22.3, 32.5, 37.2, 49.9, 56.1, 57.7, 58.3, 51.2, 42.8, 31.6, 15.9]
high_2007 = [36.5, 26.6, 43.6, 52.3, 71.5, 81.4, 80.5, 82.2, 76.0, 67.3, 46.1, 35.0]
low_2007 = [23.6, 14.0, 27.0, 36.8, 47.6, 57.7, 58.9, 61.2, 53.3, 48.5, 31.0, 23.6]
high_2014 = [28.8, 28.5, 37.0, 56.8, 69.7, 79.7, 78.5, 77.8, 74.1, 62.6, 45.3, 39.9]
low_2014 = [12.7, 14.3, 18.6, 35.5, 49.9, 58.0, 60.0, 58.6, 51.7, 45.2, 32.2, 29.1]
fig = go.Figure()
fig.add_trace(go.Scatter(x=month, y=high_2014, name='High 2014', line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=month, y=low_2014, name='Low 2014', line=dict(color='royalblue', width=4)))
fig.add_trace(go.Scatter(x=month, y=high_2007, name='High 2007', line=dict(color='firebrick', width=4, dash='dash')))
fig.add_trace(go.Scatter(x=month, y=low_2007, name='Low 2007', line=dict(color='royalblue', width=4, dash='dash')))
fig.add_trace(go.Scatter(x=month, y=high_2000, name='High 2000', line=dict(color='firebrick', width=4, dash='dot')))
fig.add_trace(go.Scatter(x=month, y=low_2000, name='Low 2000', line=dict(color='royalblue', width=4, dash='dot')))
fig.update_layout(title='Average High and Low Temperatures in New York', xaxis_title='Month', yaxis_title='Temperature (degrees F)')
fig.show()
import plotly.graph_objects as go
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=[10, 20, None, 15, 10, 5, 15, None, 20, 10, 10, 15, 25, 20, 10], name='<b>No</b> Gaps', connectgaps=True))
fig.add_trace(go.Scatter(x=x, y=[5, 15, None, 10, 5, 0, 10, None, 15, 5, 5, 10, 20, 15, 5], name='Gaps'))
fig.show()
import plotly.graph_objects as go
import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 3, 1])
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, name='linear', line_shape='linear'))
fig.add_trace(go.Scatter(x=x, y=y + 5, name='spline', text=["tweak line smoothness<br>with 'smoothing' in line object"], hoverinfo='text+name', line_shape='spline'))
fig.add_trace(go.Scatter(x=x, y=y + 10, name='vhv', line_shape='vhv'))
fig.add_trace(go.Scatter(x=x, y=y + 15, name='hvh', line_shape='hvh'))
fig.add_trace(go.Scatter(x=x, y=y + 20, name='vh', line_shape='vh'))
fig.add_trace(go.Scatter(x=x, y=y + 25, name='hv', line_shape='hv'))
fig.update_traces(hoverinfo='text+name', mode='lines+markers')
fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))
fig.show()
import plotly.graph_objects as go
import numpy as np
title = 'Main Source for News'
labels = ['Television', 'Newspaper', 'Internet', 'Radio']
colors = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']
mode_size = [8, 8, 12, 8]
line_size = [2, 2, 4, 2]
x_data = np.vstack((np.arange(2001, 2014),) * 4)
y_data = np.array([[74, 82, 80, 74, 73, 72, 74, 70, 70, 66, 66, 69], [45, 42, 50, 46, 36, 36, 34, 35, 32, 31, 31, 28], [13, 14, 20, 24, 20, 24, 24, 40, 35, 41, 43, 50], [18, 21, 18, 21, 16, 14, 13, 18, 17, 16, 19, 23]])
fig = go.Figure()
for i in range(0, 4):
    fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines', name=labels[i], line=dict(color=colors[i], width=line_size[i]), connectgaps=True))
    fig.add_trace(go.Scatter(x=[x_data[i][0], x_data[i][-1]], y=[y_data[i][0], y_data[i][-1]], mode='markers', marker=dict(color=colors[i], size=mode_size[i])))
fig.update_layout(xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=2, ticks='outside', tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')), yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False), autosize=False, margin=dict(autoexpand=False, l=100, r=20, t=110), showlegend=False, plot_bgcolor='white')
annotations = []
for (y_trace, label, color) in zip(y_data, labels, colors):
    annotations.append(dict(xref='paper', x=0.05, y=y_trace[0], xanchor='right', yanchor='middle', text=label + ' {}%'.format(y_trace[0]), font=dict(family='Arial', size=16), showarrow=False))
    annotations.append(dict(xref='paper', x=0.95, y=y_trace[11], xanchor='left', yanchor='middle', text='{}%'.format(y_trace[11]), font=dict(family='Arial', size=16), showarrow=False))
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05, xanchor='left', yanchor='bottom', text='Main Source for News', font=dict(family='Arial', size=30, color='rgb(37,37,37)'), showarrow=False))
annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1, xanchor='center', yanchor='top', text='Source: PewResearch Center & ' + 'Storytelling with data', font=dict(family='Arial', size=12, color='rgb(150,150,150)'), showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()
import plotly.graph_objects as go
import numpy as np
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_rev = x[::-1]
y1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1_upper = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y1_lower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y1_lower = y1_lower[::-1]
y2 = [5, 2.5, 5, 7.5, 5, 2.5, 7.5, 4.5, 5.5, 5]
y2_upper = [5.5, 3, 5.5, 8, 6, 3, 8, 5, 6, 5.5]
y2_lower = [4.5, 2, 4.4, 7, 4, 2, 7, 4, 5, 4.75]
y2_lower = y2_lower[::-1]
y3 = [10, 8, 6, 4, 2, 0, 2, 4, 2, 0]
y3_upper = [11, 9, 7, 5, 3, 1, 3, 5, 3, 1]
y3_lower = [9, 7, 5, 3, 1, -0.5, 1, 3, 1, -1]
y3_lower = y3_lower[::-1]
fig = go.Figure()
fig.add_trace(go.Scatter(x=x + x_rev, y=y1_upper + y1_lower, fill='toself', fillcolor='rgba(0,100,80,0.2)', line_color='rgba(255,255,255,0)', showlegend=False, name='Fair'))
fig.add_trace(go.Scatter(x=x + x_rev, y=y2_upper + y2_lower, fill='toself', fillcolor='rgba(0,176,246,0.2)', line_color='rgba(255,255,255,0)', name='Premium', showlegend=False))
fig.add_trace(go.Scatter(x=x + x_rev, y=y3_upper + y3_lower, fill='toself', fillcolor='rgba(231,107,243,0.2)', line_color='rgba(255,255,255,0)', showlegend=False, name='Ideal'))
fig.add_trace(go.Scatter(x=x, y=y1, line_color='rgb(0,100,80)', name='Fair'))
fig.add_trace(go.Scatter(x=x, y=y2, line_color='rgb(0,176,246)', name='Premium'))
fig.add_trace(go.Scatter(x=x, y=y3, line_color='rgb(231,107,243)', name='Ideal'))
fig.update_traces(mode='lines')
fig.show()
import plotly.express as px
data_canada = px.data.gapminder().query("country == 'Canada'")
fig = px.bar(data_canada, x='year', y='pop')
fig.show()
import plotly.express as px
long_df = px.data.medals_long()
fig = px.bar(long_df, x='nation', y='count', color='medal', title='Long-Form Input')
fig.show()
import plotly.express as px
wide_df = px.data.medals_wide()
fig = px.bar(wide_df, x='nation', y=['gold', 'silver', 'bronze'], title='Wide-Form Input')
fig.show()
import plotly.express as px
data = px.data.gapminder()
data_canada = data[data.country == 'Canada']
fig = px.bar(data_canada, x='year', y='pop', hover_data=['lifeExp', 'gdpPercap'], color='lifeExp', labels={'pop': 'population of Canada'}, height=400)
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.bar(df, x='sex', y='total_bill', color='time')
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.bar(df, x='sex', y='total_bill', color='smoker', barmode='group', height=400)
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.bar(df, x='sex', y='total_bill', color='smoker', barmode='group', facet_row='time', facet_col='day', category_orders={'day': ['Thur', 'Fri', 'Sat', 'Sun'], 'time': ['Lunch', 'Dinner']})
fig.show()
import plotly.graph_objects as go
animals = ['giraffes', 'orangutans', 'monkeys']
fig = go.Figure(data=[go.Bar(name='SF Zoo', x=animals, y=[20, 14, 23]), go.Bar(name='LA Zoo', x=animals, y=[12, 18, 29])])
fig.update_layout(barmode='group')
fig.show()
import plotly.graph_objects as go
animals = ['giraffes', 'orangutans', 'monkeys']
fig = go.Figure(data=[go.Bar(name='SF Zoo', x=animals, y=[20, 14, 23]), go.Bar(name='LA Zoo', x=animals, y=[12, 18, 29])])
fig.update_layout(barmode='stack')
fig.show()
import plotly.graph_objects as go
x = ['Product A', 'Product B', 'Product C']
y = [20, 14, 23]
fig = go.Figure(data=[go.Bar(x=x, y=y, hovertext=['27% market share', '24% market share', '19% market share'])])
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)
fig.update_layout(title_text='January 2013 Sales Report')
fig.show()
import plotly.graph_objects as go
x = ['Product A', 'Product B', 'Product C']
y = [20, 14, 23]
fig = go.Figure(data=[go.Bar(x=x, y=y, text=y, textposition='auto')])
fig.show()
import plotly.express as px
df = px.data.gapminder().query("continent == 'Europe' and year == 2007 and pop > 2.e6")
fig = px.bar(df, y='pop', x='country', text='pop')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
import plotly.graph_objects as go
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
fig = go.Figure()
fig.add_trace(go.Bar(x=months, y=[20, 14, 25, 16, 18, 22, 19, 15, 12, 16, 14, 17], name='Primary Product', marker_color='indianred'))
fig.add_trace(go.Bar(x=months, y=[19, 14, 22, 14, 16, 19, 15, 14, 10, 12, 12, 16], name='Secondary Product', marker_color='lightsalmon'))
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()
import plotly.graph_objects as go
colors = ['lightslategray'] * 5
colors[1] = 'crimson'
fig = go.Figure(data=[go.Bar(x=['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E'], y=[20, 14, 23, 25, 22], marker_color=colors)])
fig.update_layout(title_text='Least Used Feature')
import plotly.graph_objects as go
fig = go.Figure(data=[go.Bar(x=[1, 2, 3, 5.5, 10], y=[10, 8, 6, 4, 2], width=[0.8, 0.8, 0.8, 3.5, 4])])
fig.show()
import plotly.graph_objects as go
years = ['2016', '2017', '2018']
fig = go.Figure()
fig.add_trace(go.Bar(x=years, y=[500, 600, 700], base=[-500, -600, -700], marker_color='crimson', name='expenses'))
fig.add_trace(go.Bar(x=years, y=[300, 400, 700], base=0, marker_color='lightslategrey', name='revenue'))
fig.show()
import plotly.graph_objects as go
years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]
fig = go.Figure()
fig.add_trace(go.Bar(x=years, y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263, 350, 430, 474, 526, 488, 537, 500, 439], name='Rest of world', marker_color='rgb(55, 83, 109)'))
fig.add_trace(go.Bar(x=years, y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270, 299, 340, 403, 549, 499], name='China', marker_color='rgb(26, 118, 255)'))
fig.update_layout(title='US Export of Plastic Scrap', xaxis_tickfont_size=14, yaxis=dict(title='USD (millions)', titlefont_size=16, tickfont_size=14), legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'), barmode='group', bargap=0.15, bargroupgap=0.1)
fig.show()
import plotly.graph_objects as go
x = [1, 2, 3, 4]
fig = go.Figure()
fig.add_trace(go.Bar(x=x, y=[1, 4, 9, 16]))
fig.add_trace(go.Bar(x=x, y=[6, -8, -4.5, 8]))
fig.add_trace(go.Bar(x=x, y=[-15, -3, 4.5, -8]))
fig.add_trace(go.Bar(x=x, y=[-1, 3, -3, -4]))
fig.update_layout(barmode='relative', title_text='Relative Barmode')
fig.show()
import plotly.graph_objects as go
x = ['b', 'a', 'c', 'd']
fig = go.Figure(go.Bar(x=x, y=[2, 5, 1, 9], name='Montreal'))
fig.add_trace(go.Bar(x=x, y=[1, 4, 9, 16], name='Ottawa'))
fig.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='Toronto'))
fig.update_layout(barmode='stack', xaxis={'categoryorder': 'category ascending'})
fig.show()
import plotly.graph_objects as go
x = ['b', 'a', 'c', 'd']
fig = go.Figure(go.Bar(x=x, y=[2, 5, 1, 9], name='Montreal'))
fig.add_trace(go.Bar(x=x, y=[1, 4, 9, 16], name='Ottawa'))
fig.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='Toronto'))
fig.update_layout(barmode='stack', xaxis={'categoryorder': 'array', 'categoryarray': ['d', 'a', 'c', 'b']})
fig.show()
import plotly.graph_objects as go
x = ['b', 'a', 'c', 'd']
fig = go.Figure(go.Bar(x=x, y=[2, 5, 1, 9], name='Montreal'))
fig.add_trace(go.Bar(x=x, y=[1, 4, 9, 16], name='Ottawa'))
fig.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='Toronto'))
fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
fig.show()
import plotly.graph_objects as go
x = [['BB+', 'BB+', 'BB+', 'BB', 'BB', 'BB'], [16, 17, 18, 16, 17, 18]]
fig = go.Figure()
fig.add_bar(x=x, y=[1, 2, 3, 4, 5, 6])
fig.add_bar(x=x, y=[6, 5, 4, 3, 2, 1])
fig.update_layout(barmode='relative')
fig.show()
import plotly.express as px
df = px.data.gapminder().query('year == 2007').query("continent == 'Europe'")
df.loc[df['pop'] < 2000000.0, 'country'] = 'Other countries'
fig = px.pie(df, values='pop', names='country', title='Population of European continent')
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.pie(df, values='tip', names='day')
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.pie(df, values='tip', names='day', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.pie(df, values='tip', names='day', color='day', color_discrete_map={'Thur': 'lightcyan', 'Fri': 'cyan', 'Sat': 'royalblue', 'Sun': 'darkblue'})
fig.show()
import plotly.express as px
df = px.data.gapminder().query('year == 2007').query("continent == 'Americas'")
fig = px.pie(df, values='pop', names='country', title='Population of American continent', hover_data=['lifeExp'], labels={'lifeExp': 'life expectancy'})
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
import plotly.graph_objects as go
labels = ['Oxygen', 'Hydrogen', 'Carbon_Dioxide', 'Nitrogen']
values = [4500, 2500, 1053, 500]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()
import plotly.graph_objects as go
colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
fig = go.Figure(data=[go.Pie(labels=['Oxygen', 'Hydrogen', 'Carbon_Dioxide', 'Nitrogen'], values=[4500, 2500, 1053, 500])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()
import plotly.express as px
df = px.data.gapminder().query("continent == 'Asia'")
fig = px.pie(df, values='pop', names='country')
fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
fig.show()
import plotly.graph_objects as go
labels = ['Oxygen', 'Hydrogen', 'Carbon_Dioxide', 'Nitrogen']
values = [4500, 2500, 1053, 500]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', insidetextorientation='radial')])
fig.show()
import plotly.graph_objects as go
labels = ['Oxygen', 'Hydrogen', 'Carbon_Dioxide', 'Nitrogen']
values = [4500, 2500, 1053, 500]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
fig.show()
import plotly.graph_objects as go
labels = ['Oxygen', 'Hydrogen', 'Carbon_Dioxide', 'Nitrogen']
values = [4500, 2500, 1053, 500]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2, 0])])
fig.show()
import plotly.graph_objects as go
from plotly.subplots import make_subplots
labels = ['US', 'China', 'European Union', 'Russian Federation', 'Brazil', 'India', 'Rest of World']
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
fig.add_trace(go.Pie(labels=labels, values=[16, 15, 12, 6, 5, 4, 42], name='GHG Emissions'), 1, 1)
fig.add_trace(go.Pie(labels=labels, values=[27, 11, 25, 8, 1, 3, 25], name='CO2 Emissions'), 1, 2)
fig.update_traces(hole=0.4, hoverinfo='label+percent+name')
fig.update_layout(title_text='Global Emissions 1990-2011', annotations=[dict(text='GHG', x=0.18, y=0.5, font_size=20, showarrow=False), dict(text='CO2', x=0.82, y=0.5, font_size=20, showarrow=False)])
fig.show()
import plotly.graph_objects as go
from plotly.subplots import make_subplots
labels = ['1st', '2nd', '3rd', '4th', '5th']
night_colors = ['rgb(56, 75, 126)', 'rgb(18, 36, 37)', 'rgb(34, 53, 101)', 'rgb(36, 55, 57)', 'rgb(6, 4, 4)']
sunflowers_colors = ['rgb(177, 127, 38)', 'rgb(205, 152, 36)', 'rgb(99, 79, 37)', 'rgb(129, 180, 179)', 'rgb(124, 103, 37)']
irises_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)', 'rgb(175, 49, 35)', 'rgb(36, 73, 147)']
cafe_colors = ['rgb(146, 123, 21)', 'rgb(177, 180, 34)', 'rgb(206, 206, 40)', 'rgb(175, 51, 21)', 'rgb(35, 36, 21)']
specs = [[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]]
fig = make_subplots(rows=2, cols=2, specs=specs)
fig.add_trace(go.Pie(labels=labels, values=[38, 27, 18, 10, 7], name='Starry Night', marker_colors=night_colors), 1, 1)
fig.add_trace(go.Pie(labels=labels, values=[28, 26, 21, 15, 10], name='Sunflowers', marker_colors=sunflowers_colors), 1, 2)
fig.add_trace(go.Pie(labels=labels, values=[38, 19, 16, 14, 13], name='Irises', marker_colors=irises_colors), 2, 1)
fig.add_trace(go.Pie(labels=labels, values=[31, 24, 19, 18, 8], name='The Night Café', marker_colors=cafe_colors), 2, 2)
fig.update_traces(hoverinfo='label+percent+name', textinfo='none')
fig.update(layout_title_text='Van Gogh: 5 Most Prominent Colors Shown Proportionally', layout_showlegend=False)
fig = go.Figure(fig)
fig.show()
import plotly.graph_objects as go
from plotly.subplots import make_subplots
labels = ['Asia', 'Europe', 'Africa', 'Americas', 'Oceania']
fig = make_subplots(1, 2, specs=[[{'type': 'domain'}, {'type': 'domain'}]], subplot_titles=['1980', '2007'])
fig.add_trace(go.Pie(labels=labels, values=[4, 7, 1, 7, 0.5], scalegroup='one', name='World GDP 1980'), 1, 1)
fig.add_trace(go.Pie(labels=labels, values=[21, 15, 3, 19, 1], scalegroup='one', name='World GDP 2007'), 1, 2)
fig.update_layout(title_text='World GDP')
fig.show()
import plotly.express as px
import pandas as pd
schools = ['Brown', 'NYU', 'Notre Dame', 'Cornell', 'Tufts', 'Yale', 'Dartmouth', 'Chicago', 'Columbia', 'Duke', 'Georgetown', 'Princeton', 'U.Penn', 'Stanford', 'MIT', 'Harvard']
n_schools = len(schools)
men_salary = [72, 67, 73, 80, 76, 79, 84, 78, 86, 93, 94, 90, 92, 96, 94, 112]
women_salary = [92, 94, 100, 107, 112, 114, 114, 118, 119, 124, 131, 137, 141, 151, 152, 165]
df = pd.DataFrame(dict(school=schools * 2, salary=men_salary + women_salary, gender=['Men'] * n_schools + ['Women'] * n_schools))
fig = px.scatter(df, x='salary', y='school', color='gender', title='Gender Earnings Disparity', labels={'salary': 'Annual Salary (in thousands)'})
fig.show()
import plotly.graph_objects as go
schools = ['Brown', 'NYU', 'Notre Dame', 'Cornell', 'Tufts', 'Yale', 'Dartmouth', 'Chicago', 'Columbia', 'Duke', 'Georgetown', 'Princeton', 'U.Penn', 'Stanford', 'MIT', 'Harvard']
fig = go.Figure()
fig.add_trace(go.Scatter(x=[72, 67, 73, 80, 76, 79, 84, 78, 86, 93, 94, 90, 92, 96, 94, 112], y=schools, marker=dict(color='crimson', size=12), mode='markers', name='Women'))
fig.add_trace(go.Scatter(x=[92, 94, 100, 107, 112, 114, 114, 118, 119, 124, 131, 137, 141, 151, 152, 165], y=schools, marker=dict(color='gold', size=12), mode='markers', name='Men'))
fig.update_layout(title='Gender Earnings Disparity', xaxis_title='Annual Salary (in thousands)', yaxis_title='School')
fig.show()
import plotly.graph_objects as go
country = ['Switzerland (2011)', 'Chile (2013)', 'Japan (2014)', 'United States (2012)', 'Slovenia (2014)', 'Canada (2011)', 'Poland (2010)', 'Estonia (2015)', 'Luxembourg (2013)', 'Portugal (2011)']
voting_pop = [40, 45.7, 52, 53.6, 54.1, 54.2, 54.5, 54.7, 55.1, 56.6]
reg_voters = [49.1, 42, 52.7, 84.3, 51.7, 61.1, 55.3, 64.2, 91.1, 58.9]
fig = go.Figure()
fig.add_trace(go.Scatter(x=voting_pop, y=country, name='Percent of estimated voting age population', marker=dict(color='rgba(156, 165, 196, 0.95)', line_color='rgba(156, 165, 196, 1.0)')))
fig.add_trace(go.Scatter(x=reg_voters, y=country, name='Percent of estimated registered voters', marker=dict(color='rgba(204, 204, 204, 0.95)', line_color='rgba(217, 217, 217, 1.0)')))
fig.update_traces(mode='markers', marker=dict(line_width=1, symbol='circle', size=16))
fig.update_layout(title='Votes cast for ten lowest voting age population in OECD countries', xaxis=dict(showgrid=False, showline=True, linecolor='rgb(102, 102, 102)', tickfont_color='rgb(102, 102, 102)', showticklabels=True, dtick=10, ticks='outside', tickcolor='rgb(102, 102, 102)'), margin=dict(l=140, r=40, b=50, t=80), legend=dict(font_size=10, yanchor='middle', xanchor='right'), width=800, height=600, paper_bgcolor='white', plot_bgcolor='white', hovermode='closest')
fig.show()
import plotly.express as px
df = px.data.gapminder()
fig = px.area(df, x='year', y='pop', color='continent', line_group='country')
fig.show()
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[0, 2, 3, 5], fill='tozeroy'))
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[3, 5, 1, 7], fill='tonexty'))
fig.show()
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[0, 2, 3, 5], fill='tozeroy', mode='none'))
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[3, 5, 1, 7], fill='tonexty', mode='none'))
fig.show()
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[3, 4, 8, 3], fill=None, mode='lines', line_color='indigo'))
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[1, 6, 2, 6], fill='tonexty', mode='lines', line_color='indigo'))
fig.show()
import plotly.graph_objects as go
x = ['Winter', 'Spring', 'Summer', 'Fall']
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=[40, 60, 40, 10], hoverinfo='x+y', mode='lines', line=dict(width=0.5, color='rgb(131, 90, 241)'), stackgroup='one'))
fig.add_trace(go.Scatter(x=x, y=[20, 10, 10, 60], hoverinfo='x+y', mode='lines', line=dict(width=0.5, color='rgb(111, 231, 219)'), stackgroup='one'))
fig.add_trace(go.Scatter(x=x, y=[40, 30, 50, 30], hoverinfo='x+y', mode='lines', line=dict(width=0.5, color='rgb(184, 247, 212)'), stackgroup='one'))
fig.update_layout(yaxis_range=(0, 100))
fig.show()
import plotly.graph_objects as go
x = ['Winter', 'Spring', 'Summer', 'Fall']
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=[40, 20, 30, 40], mode='lines', line=dict(width=0.5, color='rgb(184, 247, 212)'), stackgroup='one', groupnorm='percent'))
fig.add_trace(go.Scatter(x=x, y=[50, 70, 40, 60], mode='lines', line=dict(width=0.5, color='rgb(111, 231, 219)'), stackgroup='one'))
fig.add_trace(go.Scatter(x=x, y=[70, 80, 60, 70], mode='lines', line=dict(width=0.5, color='rgb(127, 166, 238)'), stackgroup='one'))
fig.add_trace(go.Scatter(x=x, y=[100, 100, 100, 100], mode='lines', line=dict(width=0.5, color='rgb(131, 90, 241)'), stackgroup='one'))
fig.update_layout(showlegend=True, xaxis_type='category', yaxis=dict(type='linear', range=[1, 100], ticksuffix='%'))
fig.show()
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 0.5, 1, 1.5, 2], y=[0, 1, 2, 1, 0], fill='toself', fillcolor='darkviolet', hoveron='points+fills', line_color='darkviolet', text='Points + Fills', hoverinfo='text+x+y'))
fig.add_trace(go.Scatter(x=[3, 3.5, 4, 4.5, 5], y=[0, 1, 2, 1, 0], fill='toself', fillcolor='violet', hoveron='points', line_color='violet', text='Points only', hoverinfo='text+x+y'))
fig.update_layout(title='hover on <i>points</i> or <i>fill</i>', xaxis_range=[0, 5.2], yaxis_range=[0, 3])
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.bar(df, x='total_bill', y='day', orientation='h')
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.bar(df, x='total_bill', y='sex', color='day', orientation='h', hover_data=['tip', 'size'], height=400, title='Restaurant bills')
fig.show()
import plotly.graph_objects as go
fig = go.Figure(go.Bar(x=[20, 14, 23], y=['giraffes', 'orangutans', 'monkeys'], orientation='h'))
fig.show()
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Bar(y=['giraffes', 'orangutans', 'monkeys'], x=[20, 14, 23], name='SF Zoo', orientation='h', marker=dict(color='rgba(246, 78, 139, 0.6)', line=dict(color='rgba(246, 78, 139, 1.0)', width=3))))
fig.add_trace(go.Bar(y=['giraffes', 'orangutans', 'monkeys'], x=[12, 18, 29], name='LA Zoo', orientation='h', marker=dict(color='rgba(58, 71, 80, 0.6)', line=dict(color='rgba(58, 71, 80, 1.0)', width=3))))
fig.update_layout(barmode='stack')
fig.show()
import plotly.graph_objects as go
top_labels = ['Strongly<br>agree', 'Agree', 'Neutral', 'Disagree', 'Strongly<br>disagree']
colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)', 'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)', 'rgba(190, 192, 213, 1)']
x_data = [[21, 30, 21, 16, 12], [24, 31, 19, 15, 11], [27, 26, 23, 11, 13], [29, 24, 15, 18, 14]]
y_data = ['The course was effectively<br>organized', 'The course developed my<br>abilities and skills ' + 'for<br>the subject', 'The course developed ' + 'my<br>ability to think critically about<br>the subject', 'I would recommend this<br>course to a friend']
fig = go.Figure()
for i in range(0, len(x_data[0])):
    for (xd, yd) in zip(x_data, y_data):
        fig.add_trace(go.Bar(x=[xd[i]], y=[yd], orientation='h', marker=dict(color=colors[i], line=dict(color='rgb(248, 248, 249)', width=1))))
fig.update_layout(xaxis=dict(showgrid=False, showline=False, showticklabels=False, zeroline=False, domain=[0.15, 1]), yaxis=dict(showgrid=False, showline=False, showticklabels=False, zeroline=False), barmode='stack', paper_bgcolor='rgb(248, 248, 255)', plot_bgcolor='rgb(248, 248, 255)', margin=dict(l=120, r=10, t=140, b=80), showlegend=False)
annotations = []
for (yd, xd) in zip(y_data, x_data):
    annotations.append(dict(xref='paper', yref='y', x=0.14, y=yd, xanchor='right', text=str(yd), font=dict(family='Arial', size=14, color='rgb(67, 67, 67)'), showarrow=False, align='right'))
    annotations.append(dict(xref='x', yref='y', x=xd[0] / 2, y=yd, text=str(xd[0]) + '%', font=dict(family='Arial', size=14, color='rgb(248, 248, 255)'), showarrow=False))
    if yd == y_data[-1]:
        annotations.append(dict(xref='x', yref='paper', x=xd[0] / 2, y=1.1, text=top_labels[0], font=dict(family='Arial', size=14, color='rgb(67, 67, 67)'), showarrow=False))
    space = xd[0]
    for i in range(1, len(xd)):
        annotations.append(dict(xref='x', yref='y', x=space + xd[i] / 2, y=yd, text=str(xd[i]) + '%', font=dict(family='Arial', size=14, color='rgb(248, 248, 255)'), showarrow=False))
        if yd == y_data[-1]:
            annotations.append(dict(xref='x', yref='paper', x=space + xd[i] / 2, y=1.1, text=top_labels[i], font=dict(family='Arial', size=14, color='rgb(67, 67, 67)'), showarrow=False))
        space += xd[i]
fig.update_layout(annotations=annotations)
fig.show()
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
y_saving = [1.3586, 2.2623, 4.9822, 6.5097, 7.4812, 7.5133, 15.2148, 17.5205]
y_net_worth = [93453.92, 81666.57, 69889.62, 78381.53, 141395.3, 92969.02, 66090.18, 122379.3]
x = ['Japan', 'United Kingdom', 'Canada', 'Netherlands', 'United States', 'Belgium', 'Sweden', 'Switzerland']
fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.001)
fig.append_trace(go.Bar(x=y_saving, y=x, marker=dict(color='rgba(50, 171, 96, 0.6)', line=dict(color='rgba(50, 171, 96, 1.0)', width=1)), name='Household savings, percentage of household disposable income', orientation='h'), 1, 1)
fig.append_trace(go.Scatter(x=y_net_worth, y=x, mode='lines+markers', line_color='rgb(128, 0, 128)', name='Household net worth, Million USD/capita'), 1, 2)
fig.update_layout(title='Household savings & net worth for eight OECD countries', yaxis=dict(showgrid=False, showline=False, showticklabels=True, domain=[0, 0.85]), yaxis2=dict(showgrid=False, showline=True, showticklabels=False, linecolor='rgba(102, 102, 102, 0.8)', linewidth=2, domain=[0, 0.85]), xaxis=dict(zeroline=False, showline=False, showticklabels=True, showgrid=True, domain=[0, 0.42]), xaxis2=dict(zeroline=False, showline=False, showticklabels=True, showgrid=True, domain=[0.47, 1], side='top', dtick=25000), legend=dict(x=0.029, y=1.038, font_size=10), margin=dict(l=100, r=20, t=70, b=70), paper_bgcolor='rgb(248, 248, 255)', plot_bgcolor='rgb(248, 248, 255)')
annotations = []
y_s = np.round(y_saving, decimals=2)
y_nw = np.rint(y_net_worth)
for (ydn, yd, xd) in zip(y_nw, y_s, x):
    annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 20000, text='{:,}'.format(ydn) + 'M', font=dict(family='Arial', size=12, color='rgb(128, 0, 128)'), showarrow=False))
    annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3, text=str(yd) + '%', font=dict(family='Arial', size=12, color='rgb(50, 171, 96)'), showarrow=False))
annotations.append(dict(xref='paper', yref='paper', x=-0.2, y=-0.109, text='OECD "' + '(2015), Household savings (indicator), ' + 'Household net worth (indicator). doi: ' + '10.1787/cfc6f499-en (Accessed on 05 June 2015)', font=dict(family='Arial', size=10, color='rgb(150,150,150)'), showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()
import plotly.express as px
import pandas as pd
df = pd.DataFrame([dict(Task='Job A', Start='2009-01-01', Finish='2009-02-28'), dict(Task='Job B', Start='2009-03-05', Finish='2009-04-15'), dict(Task='Job C', Start='2009-02-20', Finish='2009-05-30')])
fig = px.timeline(df, x_start='Start', x_end='Finish', y='Task')
fig.update_yaxes(autorange='reversed')
fig.show()
import plotly.express as px
import pandas as pd
df = pd.DataFrame([dict(Task='Job A', Start='2009-01-01', Finish='2009-02-28', Resource='Alex'), dict(Task='Job B', Start='2009-03-05', Finish='2009-04-15', Resource='Alex'), dict(Task='Job C', Start='2009-02-20', Finish='2009-05-30', Resource='Max')])
fig = px.timeline(df, x_start='Start', x_end='Finish', y='Task', color='Resource')
fig.update_yaxes(autorange='reversed')
fig.show()
import plotly.express as px
import pandas as pd
df = pd.DataFrame([dict(Task='Job A', Start='2009-01-01', Finish='2009-02-28', Completion_pct=50), dict(Task='Job B', Start='2009-03-05', Finish='2009-04-15', Completion_pct=25), dict(Task='Job C', Start='2009-02-20', Finish='2009-05-30', Completion_pct=75)])
fig = px.timeline(df, x_start='Start', x_end='Finish', y='Task', color='Completion_pct')
fig.update_yaxes(autorange='reversed')
fig.show()
import plotly.express as px
import pandas as pd
df = pd.DataFrame([dict(Task='Job A', Start='2009-01-01', Finish='2009-02-28', Resource='Alex'), dict(Task='Job B', Start='2009-03-05', Finish='2009-04-15', Resource='Alex'), dict(Task='Job C', Start='2009-02-20', Finish='2009-05-30', Resource='Max')])
fig = px.timeline(df, x_start='Start', x_end='Finish', y='Resource', color='Resource')
fig.show()
import plotly.figure_factory as ff
df = [dict(Task='Job A', Start='2009-01-01', Finish='2009-02-28'), dict(Task='Job B', Start='2009-03-05', Finish='2009-04-15'), dict(Task='Job C', Start='2009-02-20', Finish='2009-05-30')]
fig = ff.create_gantt(df)
fig.show()
import plotly.figure_factory as ff
df = [dict(Task='Job-1', Start='2017-01-01', Finish='2017-02-02', Resource='Complete'), dict(Task='Job-1', Start='2017-02-15', Finish='2017-03-15', Resource='Incomplete'), dict(Task='Job-2', Start='2017-01-17', Finish='2017-02-17', Resource='Not Started'), dict(Task='Job-2', Start='2017-01-17', Finish='2017-02-17', Resource='Complete'), dict(Task='Job-3', Start='2017-03-10', Finish='2017-03-20', Resource='Not Started'), dict(Task='Job-3', Start='2017-04-01', Finish='2017-04-20', Resource='Not Started'), dict(Task='Job-3', Start='2017-05-18', Finish='2017-06-18', Resource='Not Started'), dict(Task='Job-4', Start='2017-01-14', Finish='2017-03-14', Resource='Complete')]
colors = {'Not Started': 'rgb(220, 0, 0)', 'Incomplete': (1, 0.9, 0.16), 'Complete': 'rgb(0, 255, 100)'}
fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True)
fig.show()
import plotly.figure_factory as ff
df = [dict(Task='Job A', Start='2009-01-01', Finish='2009-02-28', Complete=10), dict(Task='Job B', Start='2008-12-05', Finish='2009-04-15', Complete=60), dict(Task='Job C', Start='2009-02-20', Finish='2009-05-30', Complete=95)]
fig = ff.create_gantt(df, colors='Viridis', index_col='Complete', show_colorbar=True)
fig.show()
import plotly.express as px
data = dict(character=['Eve', 'Cain', 'Seth', 'Enos', 'Noam', 'Abel', 'Awan', 'Enoch', 'Azura'], parent=['', 'Eve', 'Eve', 'Seth', 'Seth', 'Eve', 'Eve', 'Awan', 'Eve'], value=[10, 14, 12, 10, 2, 6, 6, 4, 4])
fig = px.sunburst(data, names='character', parents='parent', values='value')
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.sunburst(df, path=['day', 'time', 'sex'], values='total_bill')
fig.show()
import plotly.express as px
import numpy as np
df = px.data.gapminder().query('year == 2007')
fig = px.sunburst(df, path=['continent', 'country'], values='pop', color='lifeExp', hover_data=['iso_alpha'], color_continuous_scale='RdBu', color_continuous_midpoint=np.average(df['lifeExp'], weights=df['pop']))
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.sunburst(df, path=['sex', 'day', 'time'], values='total_bill', color='day')
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.sunburst(df, path=['sex', 'day', 'time'], values='total_bill', color='time')
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.sunburst(df, path=['sex', 'day', 'time'], values='total_bill', color='time', color_discrete_map={'(?)': 'black', 'Lunch': 'gold', 'Dinner': 'darkblue'})
fig.show()
import plotly.express as px
import pandas as pd
vendors = ['A', 'B', 'C', 'D', None, 'E', 'F', 'G', 'H', None]
sectors = ['Tech', 'Tech', 'Finance', 'Finance', 'Other', 'Tech', 'Tech', 'Finance', 'Finance', 'Other']
regions = ['North', 'North', 'North', 'North', 'North', 'South', 'South', 'South', 'South', 'South']
sales = [1, 3, 2, 4, 1, 2, 2, 1, 4, 1]
df = pd.DataFrame(dict(vendors=vendors, sectors=sectors, regions=regions, sales=sales))
print(df)
fig = px.sunburst(df, path=['regions', 'sectors', 'vendors'], values='sales')
fig.show()
import plotly.graph_objects as go
fig = go.Figure(go.Sunburst(labels=['Eve', 'Cain', 'Seth', 'Enos', 'Noam', 'Abel', 'Awan', 'Enoch', 'Azura'], parents=['', 'Eve', 'Eve', 'Seth', 'Seth', 'Eve', 'Eve', 'Awan', 'Eve'], values=[10, 14, 12, 10, 2, 6, 6, 4, 4]))
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
fig.show()
import plotly.graph_objects as go
fig = go.Figure(go.Sunburst(ids=['North America', 'Europe', 'Australia', 'North America - Football', 'Soccer', 'North America - Rugby', 'Europe - Football', 'Rugby', 'Europe - American Football', 'Australia - Football', 'Association', 'Australian Rules', 'Autstralia - American Football', 'Australia - Rugby', 'Rugby League', 'Rugby Union'], labels=['North<br>America', 'Europe', 'Australia', 'Football', 'Soccer', 'Rugby', 'Football', 'Rugby', 'American<br>Football', 'Football', 'Association', 'Australian<br>Rules', 'American<br>Football', 'Rugby', 'Rugby<br>League', 'Rugby<br>Union'], parents=['', '', '', 'North America', 'North America', 'North America', 'Europe', 'Europe', 'Europe', 'Australia', 'Australia - Football', 'Australia - Football', 'Australia - Football', 'Australia - Football', 'Australia - Rugby', 'Australia - Rugby']))
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
fig.show()
import plotly.graph_objects as go
fig = go.Figure(go.Sunburst(labels=['Eve', 'Cain', 'Seth', 'Enos', 'Noam', 'Abel', 'Awan', 'Enoch', 'Azura'], parents=['', 'Eve', 'Eve', 'Seth', 'Seth', 'Eve', 'Eve', 'Awan', 'Eve'], values=[65, 14, 12, 10, 2, 6, 6, 4, 4], branchvalues='total'))
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
fig.show()
import plotly.graph_objects as go
import pandas as pd
df1 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/sunburst-coffee-flavors-complete.csv')
df2 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/coffee-flavors.csv')
fig = go.Figure()
fig.add_trace(go.Sunburst(ids=df1.ids, labels=df1.labels, parents=df1.parents, domain=dict(column=0)))
fig.add_trace(go.Sunburst(ids=df2.ids, labels=df2.labels, parents=df2.parents, domain=dict(column=1), maxdepth=2))
fig.update_layout(grid=dict(columns=2, rows=1), margin=dict(t=0, l=0, r=0, b=0))
fig.show()
import plotly.graph_objects as go
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/coffee-flavors.csv')
fig = go.Figure()
fig.add_trace(go.Sunburst(ids=df.ids, labels=df.labels, parents=df.parents, domain=dict(column=1), maxdepth=2, insidetextorientation='radial'))
fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
fig.show()
import plotly.graph_objects as go
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/sunburst-coffee-flavors-complete.csv')
fig = go.Figure(go.Sunburst(ids=df.ids, labels=df.labels, parents=df.parents))
fig.update_layout(uniformtext=dict(minsize=10, mode='hide'))
fig.show()
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/sales_success.csv')
print(df.head())
levels = ['salesperson', 'county', 'region']
color_columns = ['sales', 'calls']
value_column = 'calls'

def build_hierarchical_dataframe(df, levels, value_column, color_columns=None):
    """
    Build a hierarchy of levels for Sunburst or Treemap charts.

    Levels are given starting from the bottom to the top of the hierarchy,
    ie the last level corresponds to the root.
    """
    df_all_trees = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])
    for (i, level) in enumerate(levels):
        df_tree = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])
        dfg = df.groupby(levels[i:]).sum()
        dfg = dfg.reset_index()
        df_tree['id'] = dfg[level].copy()
        if i < len(levels) - 1:
            df_tree['parent'] = dfg[levels[i + 1]].copy()
        else:
            df_tree['parent'] = 'total'
        df_tree['value'] = dfg[value_column]
        df_tree['color'] = dfg[color_columns[0]] / dfg[color_columns[1]]
        df_all_trees = df_all_trees.append(df_tree, ignore_index=True)
    total = pd.Series(dict(id='total', parent='', value=df[value_column].sum(), color=df[color_columns[0]].sum() / df[color_columns[1]].sum()))
    df_all_trees = df_all_trees.append(total, ignore_index=True)
    return df_all_trees
df_all_trees = build_hierarchical_dataframe(df, levels, value_column, color_columns)
average_score = df['sales'].sum() / df['calls'].sum()
fig = make_subplots(1, 2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
fig.add_trace(go.Sunburst(labels=df_all_trees['id'], parents=df_all_trees['parent'], values=df_all_trees['value'], branchvalues='total', marker=dict(colors=df_all_trees['color'], colorscale='RdBu', cmid=average_score), hovertemplate='<b>%{label} </b> <br> Sales: %{value}<br> Success rate: %{color:.2f}', name=''), 1, 1)
fig.add_trace(go.Sunburst(labels=df_all_trees['id'], parents=df_all_trees['parent'], values=df_all_trees['value'], branchvalues='total', marker=dict(colors=df_all_trees['color'], colorscale='RdBu', cmid=average_score), hovertemplate='<b>%{label} </b> <br> Sales: %{value}<br> Success rate: %{color:.2f}', maxdepth=2), 1, 2)
fig.update_layout(margin=dict(t=10, b=10, r=10, l=10))
fig.show()
import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores']), cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))])
fig.show()
import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores'], line_color='darkslategray', fill_color='lightskyblue', align='left'), cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]], line_color='darkslategray', fill_color='lightcyan', align='left'))])
fig.update_layout(width=500, height=300)
fig.show()
import plotly.graph_objects as go
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')
fig = go.Figure(data=[go.Table(header=dict(values=list(df.columns), fill_color='paleturquoise', align='left'), cells=dict(values=[df.Rank, df.State, df.Postal, df.Population], fill_color='lavender', align='left'))])
fig.show()
import plotly.graph_objects as go
values = [['Salaries', 'Office', 'Merchandise', 'Legal', '<b>TOTAL<br>EXPENSES</b>'], ['Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad', 'Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad', 'Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad', 'Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad', 'Lorem ipsum dolor sit amet, tollit discere inermis pri ut. Eos ea iusto timeam, an prima laboramus vim. Id usu aeterno adversarium, summo mollis timeam vel ad']]
fig = go.Figure(data=[go.Table(columnorder=[1, 2], columnwidth=[80, 400], header=dict(values=[['<b>EXPENSES</b><br>as of July 2017'], ['<b>DESCRIPTION</b>']], line_color='darkslategray', fill_color='royalblue', align=['left', 'center'], font=dict(color='white', size=12), height=40), cells=dict(values=values, line_color='darkslategray', fill=dict(color=['paleturquoise', 'white']), align=['left', 'center'], font_size=12, height=30))])
fig.show()
import plotly.graph_objects as go
headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'
fig = go.Figure(data=[go.Table(header=dict(values=['<b>EXPENSES</b>', '<b>Q1</b>', '<b>Q2</b>', '<b>Q3</b>', '<b>Q4</b>'], line_color='darkslategray', fill_color=headerColor, align=['left', 'center'], font=dict(color='white', size=12)), cells=dict(values=[['Salaries', 'Office', 'Merchandise', 'Legal', '<b>TOTAL</b>'], [1200000, 20000, 80000, 2000, 12120000], [1300000, 20000, 70000, 2000, 130902000], [1300000, 20000, 120000, 2000, 131222000], [1400000, 20000, 90000, 2000, 14102000]], line_color='darkslategray', fill_color=[[rowOddColor, rowEvenColor, rowOddColor, rowEvenColor, rowOddColor] * 5], align=['left', 'center'], font=dict(color='darkslategray', size=11)))])
fig.show()
import plotly.graph_objects as go
import pandas as pd
colors = ['rgb(239, 243, 255)', 'rgb(189, 215, 231)', 'rgb(107, 174, 214)', 'rgb(49, 130, 189)', 'rgb(8, 81, 156)']
data = {'Year': [2010, 2011, 2012, 2013, 2014], 'Color': colors}
df = pd.DataFrame(data)
fig = go.Figure(data=[go.Table(header=dict(values=['Color', '<b>YEAR</b>'], line_color='white', fill_color='white', align='center', font=dict(color='black', size=12)), cells=dict(values=[df.Color, df.Year], line_color=[df.Color], fill_color=[df.Color], align='center', font=dict(color='black', size=11)))])
fig.show()
import plotly.graph_objects as go
from plotly.colors import n_colors
import numpy as np
np.random.seed(1)
colors = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 9, colortype='rgb')
a = np.random.randint(low=0, high=9, size=10)
b = np.random.randint(low=0, high=9, size=10)
c = np.random.randint(low=0, high=9, size=10)
fig = go.Figure(data=[go.Table(header=dict(values=['<b>Column A</b>', '<b>Column B</b>', '<b>Column C</b>'], line_color='white', fill_color='white', align='center', font=dict(color='black', size=12)), cells=dict(values=[a, b, c], line_color=[np.array(colors)[a], np.array(colors)[b], np.array(colors)[c]], fill_color=[np.array(colors)[a], np.array(colors)[b], np.array(colors)[c]], align='center', font=dict(color='white', size=11)))])
fig.show()
import plotly.graph_objects as go
fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], color='blue'), link=dict(source=[0, 1, 0, 2, 3, 3], target=[2, 3, 3, 4, 4, 5], value=[8, 4, 2, 8, 4, 2]))])
fig.update_layout(title_text='Basic Sankey Diagram', font_size=10)
fig.show()
import plotly.graph_objects as go
import urllib, json
url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())
opacity = 0.4
data['data'][0]['node']['color'] = ['rgba(255,0,255, 0.8)' if color == 'magenta' else color for color in data['data'][0]['node']['color']]
data['data'][0]['link']['color'] = [data['data'][0]['node']['color'][src].replace('0.8', str(opacity)) for src in data['data'][0]['link']['source']]
fig = go.Figure(data=[go.Sankey(valueformat='.0f', valuesuffix='TWh', node=dict(pad=15, thickness=15, line=dict(color='black', width=0.5), label=data['data'][0]['node']['label'], color=data['data'][0]['node']['color']), link=dict(source=data['data'][0]['link']['source'], target=data['data'][0]['link']['target'], value=data['data'][0]['link']['value'], label=data['data'][0]['link']['label'], color=data['data'][0]['link']['color']))])
fig.update_layout(title_text="Energy forecast for 2050<br>Source: Department of Energy & Climate Change, Tom Counsell via <a href='https://bost.ocks.org/mike/sankey/'>Mike Bostock</a>", font_size=10)
fig.show()
import plotly.graph_objects as go
import urllib, json
url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())
fig = go.Figure(data=[go.Sankey(valueformat='.0f', valuesuffix='TWh', node=dict(pad=15, thickness=15, line=dict(color='black', width=0.5), label=data['data'][0]['node']['label'], color=data['data'][0]['node']['color']), link=dict(source=data['data'][0]['link']['source'], target=data['data'][0]['link']['target'], value=data['data'][0]['link']['value'], label=data['data'][0]['link']['label']))])
fig.update_layout(hovermode='x', title="Energy forecast for 2050<br>Source: Department of Energy & Climate Change, Tom Counsell via <a href='https://bost.ocks.org/mike/sankey/'>Mike Bostock</a>", font=dict(size=10, color='white'), plot_bgcolor='black', paper_bgcolor='black')
fig.show()
import plotly.graph_objects as go
fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], customdata=['Long name A1', 'Long name A2', 'Long name B1', 'Long name B2', 'Long name C1', 'Long name C2'], hovertemplate='Node %{customdata} has total value %{value}<extra></extra>', color='blue'), link=dict(source=[0, 1, 0, 2, 3, 3], target=[2, 3, 3, 4, 4, 5], value=[8, 4, 2, 8, 4, 2], customdata=['q', 'r', 's', 't', 'u', 'v'], hovertemplate='Link from node %{source.customdata}<br />' + 'to node%{target.customdata}<br />has value %{value}' + '<br />and data %{customdata}<extra></extra>'))])
fig.update_layout(title_text='Basic Sankey Diagram', font_size=10)
fig.show()
import plotly.graph_objects as go
fig = go.Figure(go.Sankey(arrangement='snap', node={'label': ['A', 'B', 'C', 'D', 'E', 'F'], 'x': [0.2, 0.1, 0.5, 0.7, 0.3, 0.5], 'y': [0.7, 0.5, 0.2, 0.4, 0.2, 0.3], 'pad': 10}, link={'source': [0, 0, 1, 2, 5, 4, 3, 5], 'target': [5, 3, 4, 3, 0, 2, 2, 3], 'value': [1, 2, 1, 1, 1, 1, 1, 2]}))
fig.show()
import plotly.express as px
fig = px.treemap(names=['Eve', 'Cain', 'Seth', 'Enos', 'Noam', 'Abel', 'Awan', 'Enoch', 'Azura'], parents=['', 'Eve', 'Eve', 'Seth', 'Seth', 'Eve', 'Eve', 'Awan', 'Eve'])
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.treemap(df, path=['day', 'time', 'sex'], values='total_bill')
fig.show()
import plotly.express as px
import numpy as np
df = px.data.gapminder().query('year == 2007')
df['world'] = 'world'
fig = px.treemap(df, path=['world', 'continent', 'country'], values='pop', color='lifeExp', hover_data=['iso_alpha'], color_continuous_scale='RdBu', color_continuous_midpoint=np.average(df['lifeExp'], weights=df['pop']))
fig.show()
import plotly.express as px
df = px.data.tips()
df['all'] = 'all'
fig = px.treemap(df, path=['all', 'sex', 'day', 'time'], values='total_bill', color='day')
fig.show()
import plotly.express as px
df = px.data.tips()
df['all'] = 'all'
fig = px.treemap(df, path=['all', 'sex', 'day', 'time'], values='total_bill', color='time')
fig.show()
import plotly.express as px
df = px.data.tips()
fig = px.treemap(df, path=['sex', 'day', 'time'], values='total_bill', color='time', color_discrete_map={'(?)': 'black', 'Lunch': 'gold', 'Dinner': 'darkblue'})
fig.show()
import plotly.express as px
import pandas as pd
vendors = ['A', 'B', 'C', 'D', None, 'E', 'F', 'G', 'H', None]
sectors = ['Tech', 'Tech', 'Finance', 'Finance', 'Other', 'Tech', 'Tech', 'Finance', 'Finance', 'Other']
regions = ['North', 'North', 'North', 'North', 'North', 'South', 'South', 'South', 'South', 'South']
sales = [1, 3, 2, 4, 1, 2, 2, 1, 4, 1]
df = pd.DataFrame(dict(vendors=vendors, sectors=sectors, regions=regions, sales=sales))
df['all'] = 'all'
print(df)
fig = px.treemap(df, path=['all', 'regions', 'sectors', 'vendors'], values='sales')
fig.show()
import plotly.express as px
import pandas as pd
import numpy as np
np.random.seed(1)
N = 100000
df = pd.DataFrame(dict(x=np.random.randn(N), y=np.random.randn(N)))
fig = px.scatter(df, x='x', y='y', render_mode='webgl')
fig.update_traces(marker_line=dict(width=1, color='DarkSlateGray'))
fig.show()
import plotly.graph_objects as go
import numpy as np
N = 100000
fig = go.Figure()
fig.add_trace(go.Scattergl(x=np.random.randn(N), y=np.random.randn(N), mode='markers', marker=dict(line=dict(width=1, color='DarkSlateGrey'))))
fig.show()
import plotly.graph_objects as go
import numpy as np
N = 1000000
fig = go.Figure()
fig.add_trace(go.Scattergl(x=np.random.randn(N), y=np.random.randn(N), mode='markers', marker=dict(line=dict(width=1, color='DarkSlateGrey'))))
fig.show()
import plotly.graph_objects as go
import numpy as np
fig = go.Figure()
trace_num = 10
point_num = 5000
for i in range(trace_num):
    fig.add_trace(go.Scattergl(x=np.linspace(0, 1, point_num), y=np.random.randn(point_num) + i * 5))
fig.update_layout(showlegend=False)
fig.show()
import plotly.figure_factory as ff
data_matrix = [['Country', 'Year', 'Population'], ['United States', 2000, 282200000], ['Canada', 2000, 27790000], ['United States', 2005, 295500000], ['Canada', 2005, 32310000], ['United States', 2010, 309000000], ['Canada', 2010, 34000000]]
fig = ff.create_table(data_matrix)
fig.show()
import plotly.figure_factory as ff
data_matrix = [['User', 'Language', 'Chart Type', '# of Views'], ['<a href="https://plotly.com/~empet/folder/home">empet</a>', '<a href="https://plotly.com/python/">Python</a>', '<a href="https://plotly.com/~empet/8614/">Network Graph</a>', 298], ['<a href="https://plotly.com/~Grondo/folder/home">Grondo</a>', '<a href="https://plotly.com/matlab/">Matlab</a>', '<a href="https://plotly.com/~Grondo/42/">Subplots</a>', 356], ['<a href="https://plotly.com/~Dreamshot/folder/home">Dreamshot</a>', '<a href="https://help.plot.ly/tutorials/">Web App</a>', '<a href="https://plotly.com/~Dreamshot/6575/_2014-us-city-populations/">Bubble Map</a>', 262], ['<a href="https://plotly.com/~FiveThirtyEight/folder/home">FiveThirtyEight</a>', '<a href="https://help.plot.ly/tutorials/">Web App</a>', '<a href="https://plotly.com/~FiveThirtyEight/30/">Scatter</a>', 692], ['<a href="https://plotly.com/~cpsievert/folder/home">cpsievert</a>', '<a href="https://plotly.com/r/">R</a>', '<a href="https://plotly.com/~cpsievert/1130/">Surface</a>', 302]]
fig = ff.create_table(data_matrix)
fig.show()
import plotly.figure_factory as ff
data_matrix = [['Name', 'Equation'], ['Pythagorean Theorem', '$a^{2}+b^{2}=c^{2}$'], ["Euler's Formula", '$F-E+V=2$'], ['The Origin of Complex Numbers', '$i^{2}=-1$'], ["Einstein's Theory of Relativity", '$E=m c^{2}$']]
fig = ff.create_table(data_matrix)
fig.show()
import plotly.figure_factory as ff
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')
df_sample = df[100:120]
fig = ff.create_table(df_sample)
fig.show()
import plotly.figure_factory as ff
data_matrix = [['Country', 'Year', 'Population'], ['United States', 2000, 282200000], ['Canada', 2000, 27790000], ['United States', 2005, 295500000], ['Canada', 2005, 32310000], ['United States', 2010, 309000000], ['Canada', 2010, 34000000]]
fig = ff.create_table(data_matrix, height_constant=20)
fig.show()
import plotly.figure_factory as ff
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')
df_sample = df[400:410]
colorscale = [[0, '#4d004c'], [0.5, '#f2e5ff'], [1, '#ffffff']]
fig = ff.create_table(df_sample, colorscale=colorscale)
fig.show()
import plotly.figure_factory as ff
text = [['Team', 'Rank'], ['A', 1], ['B', 2], ['C', 3], ['D', 4], ['E', 5], ['F', 6]]
colorscale = [[0, '#272D31'], [0.5, '#ffffff'], [1, '#ffffff']]
font = ['#FCFCFC', '#00EE00', '#008B00', '#004F00', '#660000', '#CD0000', '#FF3030']
fig = ff.create_table(text, colorscale=colorscale, font_colors=font)
fig.layout.width = 250
fig.show()
import plotly.figure_factory as ff
data_matrix = [['Country', 'Year', 'Population'], ['United States', 2000, 282200000], ['Canada', 2000, 27790000], ['United States', 2005, 295500000], ['Canada', 2005, 32310000], ['United States', 2010, 309000000], ['Canada', 2010, 34000000]]
fig = ff.create_table(data_matrix, index=True)
for i in range(len(fig.layout.annotations)):
    fig.layout.annotations[i].font.size = 20
fig.show()
import plotly.graph_objs as go
import plotly.figure_factory as ff
table_data = [['Team', 'Wins', 'Losses', 'Ties'], ['Montréal<br>Canadiens', 18, 4, 0], ['Dallas Stars', 18, 5, 0], ['NY Rangers', 16, 5, 0], ['Boston<br>Bruins', 13, 8, 0], ['Chicago<br>Blackhawks', 13, 8, 0], ['LA Kings', 13, 8, 0], ['Ottawa<br>Senators', 12, 5, 0]]
fig = ff.create_table(table_data, height_constant=60)
teams = ['Montréal Canadiens', 'Dallas Stars', 'NY Rangers', 'Boston Bruins', 'Chicago Blackhawks', 'LA Kings', 'Ottawa Senators']
GFPG = [3.54, 3.48, 3.0, 3.27, 2.83, 2.45, 3.18]
GAPG = [2.17, 2.57, 2.0, 2.91, 2.57, 2.14, 2.77]
fig.add_trace(go.Scatter(x=teams, y=GFPG, marker=dict(color='#0099ff'), name='Goals For<br>Per Game', xaxis='x2', yaxis='y2'))
fig.add_trace(go.Scatter(x=teams, y=GAPG, marker=dict(color='#404040'), name='Goals Against<br>Per Game', xaxis='x2', yaxis='y2'))
fig.update_layout(title_text='2016 Hockey Stats', margin={'t': 50, 'b': 100}, xaxis={'domain': [0, 0.5]}, xaxis2={'domain': [0.6, 1.0]}, yaxis2={'anchor': 'x2', 'title': 'Goals'})
fig.show()
import plotly.graph_objs as go
import plotly.figure_factory as ff
table_data = [['Team', 'Wins', 'Losses', 'Ties'], ['Montréal<br>Canadiens', 18, 4, 0], ['Dallas Stars', 18, 5, 0], ['NY Rangers', 16, 5, 0], ['Boston<br>Bruins', 13, 8, 0], ['Chicago<br>Blackhawks', 13, 8, 0], ['Ottawa<br>Senators', 12, 5, 0]]
fig = ff.create_table(table_data, height_constant=60)
teams = ['Montréal Canadiens', 'Dallas Stars', 'NY Rangers', 'Boston Bruins', 'Chicago Blackhawks', 'Ottawa Senators']
GFPG = [3.54, 3.48, 3.0, 3.27, 2.83, 3.18]
GAPG = [2.17, 2.57, 2.0, 2.91, 2.57, 2.77]
fig.add_trace(go.Bar(x=teams, y=GFPG, xaxis='x2', yaxis='y2', marker=dict(color='#0099ff'), name='Goals For<br>Per Game'))
fig.add_trace(go.Bar(x=teams, y=GAPG, xaxis='x2', yaxis='y2', marker=dict(color='#404040'), name='Goals Against<br>Per Game'))
fig.update_layout(title_text='2016 Hockey Stats', height=800, margin={'t': 75, 'l': 50}, yaxis={'domain': [0, 0.45]}, xaxis2={'anchor': 'y2'}, yaxis2={'domain': [0.6, 1], 'anchor': 'x2', 'title': 'Goals'})
fig.show()