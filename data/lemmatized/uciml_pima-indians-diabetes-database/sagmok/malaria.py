import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe()
df.sum().unique()
x = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values
df.shape
import seaborn as sns
corrmat = df.corr()
top_corr_features = corrmat.index
pass
pass
import matplotlib.pyplot as plt
import seaborn as sns
pass
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
classifer = RandomForestClassifier(n_estimators=10, random_state=0)