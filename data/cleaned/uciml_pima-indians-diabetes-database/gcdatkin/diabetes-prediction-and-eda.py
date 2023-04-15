import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, vmin=-1.0, vmax=1.0, cmap='mako')

age_ct = pd.crosstab(pd.qcut(data['Age'], q=4, labels=['Youngest', 'Younger', 'Older', 'Oldest']), data['Outcome'])
age_ct_avgs = age_ct[1] / (age_ct[0] + age_ct[1])
age_ct = pd.concat([age_ct, age_ct_avgs], axis=1)
age_ct.columns = ['Negative', 'Positive', '% Positive']
age_ct
scaler = StandardScaler()
scaled_columns = data.iloc[:, :-1]
scaled_columns = pd.DataFrame(scaler.fit_transform(scaled_columns), columns=scaled_columns.columns)
plt.figure(figsize=(18, 10))
for column in scaled_columns.columns:
    sns.kdeplot(scaled_columns[column], shade=True)

y = data.loc[:, 'Outcome']
X = data.drop('Outcome', axis=1)
X = scaler.fit_transform(X)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=24)
log_model = LogisticRegression()
svm_model = SVC(C=1.0)
ann_model = MLPClassifier(hidden_layer_sizes=(16, 16))