import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
df.info()
pass
pass
pass
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()