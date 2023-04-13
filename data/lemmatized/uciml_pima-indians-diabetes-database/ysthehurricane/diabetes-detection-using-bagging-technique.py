import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.isnull().sum()
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
X = df.drop('Outcome', axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
kfold = model_selection.KFold(n_splits=3)
base_cls = DecisionTreeClassifier()
num_trees = 400
model = BaggingClassifier(base_estimator=base_cls, n_estimators=num_trees)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print('accuracy :')
print(results.mean())