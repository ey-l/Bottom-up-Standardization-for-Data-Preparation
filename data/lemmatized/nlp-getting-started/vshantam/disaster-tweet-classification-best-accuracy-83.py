import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv')[['text', 'target']]
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
df_train.info()
df_train.describe()
y_train = df_train['target'].unique()
y_train
import plotly.express as px
train_filtered = df_train
fig = px.violin(train_filtered, x='target', points='outliers')
fig.update_layout(title='target  violin', xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)), yaxis=dict(showgrid=True), legend=dict(orientation='v'), paper_bgcolor='#FFFFFF')
fig.show(renderer='iframe')
import matplotlib.pyplot as plt
import seaborn as sns
locations_vc = df_train['target'].value_counts()
sns.displot(y=locations_vc.index, x=locations_vc)
plt.title('target')
import plotly.express as px
fig = px.histogram(df_train, x='target', color='target', nbins=10, histnorm='density', histfunc='count')
fig.update_layout(title='label histogram', xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)), yaxis=dict(showgrid=True), legend=dict(orientation='v'), barmode='stack', paper_bgcolor='#FFFFFF')
fig.show(renderer='iframe')
import plotly.express as px
train_filtered = df_train.head(500)
fig = px.histogram(train_filtered, x='text', color='target', histfunc='count')
fig.update_layout(title='text (first 500 rows) histogram', xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)), yaxis=dict(showgrid=True), legend=dict(orientation='v'), barmode='group', paper_bgcolor='#FFFFFF')
fig.show(renderer='iframe')
import plotly.express as px
from mitosheet import *
register_analysis('id-zfdzjgornl')
df_train.insert(1, 'char_count', 0)
df_train['char_count'] = LEN(df_train['text'])
df_train['token_count'] = df_train[['text']].apply(lambda x: list(x.str.split(' '))).apply(lambda x: x.str.len())
train_filtered = df_train
fig = px.density_contour(train_filtered, x='target', y='char_count')
fig.update_layout(title='label, char_count density contour', xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)), yaxis=dict(showgrid=True), legend=dict(orientation='v'), paper_bgcolor='#FFFFFF')
fig.show(renderer='iframe')
sns.kdeplot(df_train[df_train['target'] == 1]['text'].str.len(), shade=True, color='red')
sns.kdeplot(df_train[df_train['target'] == 0]['text'].str.len(), shade=True, color='blue')
import plotly.express as px
fig = px.density_heatmap(df_train, x='target', y='char_count')
fig.update_layout(title='label, char_count density heatmap', xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)), yaxis=dict(showgrid=True), legend=dict(orientation='v'), paper_bgcolor='#FFFFFF')
fig.show(renderer='iframe')
import plotly.express as px
fig = px.bar(df_train, x='target', y='char_count', color='target')
fig.update_layout(title='label, char_count bar chart', xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)), yaxis=dict(showgrid=True), legend=dict(orientation='v'), barmode='overlay', barnorm='percent', paper_bgcolor='#FFFFFF')
fig.show(renderer='iframe')
import plotly.express as px
train_filtered = df_train
fig = px.density_contour(train_filtered, x='target', y='token_count')
fig.update_layout(title='label token_count density contour', xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)), yaxis=dict(showgrid=True), legend=dict(orientation='v'), paper_bgcolor='#FFFFFF')
fig.show(renderer='iframe')
import plotly.express as px
fig = px.density_heatmap(df_train, x='target', y='token_count')
fig.update_layout(title='target, token_count density heatmap', xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)), yaxis=dict(showgrid=True), legend=dict(orientation='v'), paper_bgcolor='#FFFFFF')
fig.show(renderer='iframe')
import plotly.express as px
fig = px.bar(df_train, x='target', y='token_count', color='target')
fig.update_layout(title='target, token_count bar chart', xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)), yaxis=dict(showgrid=True), legend=dict(orientation='v'), barmode='overlay', barnorm='percent', paper_bgcolor='#FFFFFF')
fig.show(renderer='iframe')
import plotly.express as px
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
pie_df = pd.DataFrame({'category': sorted(['True', 'False']), 'values': [x / df_train.shape[0] for x in df_train['target'].value_counts().tolist()]})
fig = px.pie(pie_df, values='values', names='category', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
corr_df = df_train[['target', 'token_count']]