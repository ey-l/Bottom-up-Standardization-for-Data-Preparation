import pandas as pd
colnames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', names=colnames, skiprows=1)
X = df.drop('class', axis=1)
y = df['class']
df.head()
df.info()
df.describe()
from matplotlib import pyplot as plt
import seaborn as sns
sns.pairplot(df, diag_kind='kde')
from sklearn.ensemble import GradientBoostingClassifier
dt = GradientBoostingClassifier()