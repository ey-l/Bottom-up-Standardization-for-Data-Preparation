import pandas as pd
from matplotlib import pyplot as plt
from sklearn import ensemble, metrics, preprocessing, neighbors
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
data_train = df.sample(frac=0.8, random_state=1)
data_test = df.drop(data_train.index)
print('Taille data_train:', len(data_train))
print('Taille data_test:', len(data_test))
X_train = data_train.drop(['Outcome'], axis=1)
y_train = data_train['Outcome']
X_test = data_test.drop(['Outcome'], axis=1)
y_test = data_test['Outcome']
X_train.head(3)
rf = ensemble.RandomForestClassifier()