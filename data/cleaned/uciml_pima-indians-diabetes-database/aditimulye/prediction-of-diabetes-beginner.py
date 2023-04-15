import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
df.info()
sns.pairplot(df, palette='viridis')
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), cmap='viridis', annot=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()