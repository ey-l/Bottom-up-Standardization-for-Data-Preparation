import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.describe()
data.info()
sns.set_style(style='whitegrid')
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.countplot(x='Outcome', data=data)
sns.distplot(data['BloodPressure'], color='indigo')
sns.jointplot(x='BloodPressure', y='Age', data=data, kind='kde', color='red')
sns.countplot(x='Pregnancies', data=data, hue='Outcome', palette='plasma', saturation=10.75)
sns.pairplot(data, hue='Outcome')
from sklearn.preprocessing import StandardScaler
scaled = StandardScaler()