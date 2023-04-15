import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
Df
Df.info()
sns.pairplot(Df)
sns.heatmap(Df.isnull(), yticklabels=False, cmap='viridis')
corr = Df.corr()
sns.heatmap(corr, annot=True)
Df.info()
Df = Df.drop('BloodPressure', axis=1)
Df
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X = Df.drop('Outcome', axis=1)
y = Df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()