import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')

df.info()
for i in range(0, len(df.columns), 5):
    sns.pairplot(data=df, x_vars=df.columns[i:i + 5], y_vars=['Outcome'])
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
iqr = q3 - q1
print(iqr)
df = df[~((df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)).any(axis=1)]
for i in range(0, len(df.columns), 5):
    sns.pairplot(data=df, x_vars=df.columns[i:i + 5], y_vars=['Outcome'])
outcome = df['Outcome']
features = df.drop('Outcome', axis=1)
(features_train, features_test, labels_train, labels_test) = train_test_split(features, outcome, test_size=0.1)
clf = KNeighborsClassifier()