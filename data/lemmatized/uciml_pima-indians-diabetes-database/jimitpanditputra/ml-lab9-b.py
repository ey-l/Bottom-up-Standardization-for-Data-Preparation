import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()