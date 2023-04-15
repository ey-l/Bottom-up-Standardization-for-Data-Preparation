import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
for feature in df.columns:
    plt.title(feature)
    plt.hist(df[feature], bins=15)

import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(df.drop('Outcome', axis=1), df['Outcome'])
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
model = cross_val_score(DecisionTreeClassifier(), df.drop('Outcome', axis=1), df['Outcome'], cv=5)
model
model.mean()
from sklearn.ensemble import BaggingClassifier
bagging_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, oob_score=True, max_samples=0.8, random_state=0)
'\nbase_estimator : model will be used on subgroups to be trained\nn_estimator : how many of them\noob_score: use or not to use oob samples as testing samples\nmax_samples: how much samples these sub groups should contain from the original dataset\n\n'