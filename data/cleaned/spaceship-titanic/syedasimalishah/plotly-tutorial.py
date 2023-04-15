import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pyo.init_notebook_mode()
df1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
df1.head()
px.pie(df1, names='HomePlanet')
px.histogram(df1, x='Age', nbins=4)
px.box(df1, y=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
px.box(df1, y='Age')
px.scatter(df1, x='Age', y='Spa')
px.bar(df1, x='Age', y='Spa', color='HomePlanet')