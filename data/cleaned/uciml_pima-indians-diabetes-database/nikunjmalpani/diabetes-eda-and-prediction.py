import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
df.info()
sns.color_palette('hls', 8)
sns.pairplot(df, hue='Outcome')
sns.heatmap(df.corr(), cmap='viridis')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()