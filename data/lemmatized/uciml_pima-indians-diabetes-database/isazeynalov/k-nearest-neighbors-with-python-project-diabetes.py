import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe()
import missingno as msno
msno.bar(df)
pass
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()