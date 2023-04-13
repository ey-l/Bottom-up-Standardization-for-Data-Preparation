import pandas as pd
from sklearn import tree
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', header=0)
data.head()
data.info()
data['BMI'] = data['BMI'].astype(int)
data['DiabetesPedigreeFunction'] = data['DiabetesPedigreeFunction'].astype(int)
features = list(data.columns[:8])
features
y = data['Outcome']
x = data[features]
Tree = tree.DecisionTreeClassifier()