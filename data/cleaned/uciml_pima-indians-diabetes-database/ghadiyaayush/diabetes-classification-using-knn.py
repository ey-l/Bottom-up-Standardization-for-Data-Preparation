import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.describe()
sns.pairplot(df, hue='Outcome')
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
sns.set_theme(style='whitegrid')
sns.boxplot(x='Age', data=df, palette='Set3')
plt.title('Age Distribution')
fig = plt.figure(figsize=(15, 20))
ax = fig.gca()
df.hist(ax=ax)
df.Outcome.value_counts().plot(kind='bar')
plt.xlabel('Diabetes or Not')
plt.ylabel('Count')
plt.title('Outcome ')
X = df.drop('Outcome', axis=1)
X.head()
y = df['Outcome']
y.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', p=2)