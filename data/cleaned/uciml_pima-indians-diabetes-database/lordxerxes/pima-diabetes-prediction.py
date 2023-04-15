import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info()
data.isnull().sum()
corr = data.corr()
corr
import seaborn as sns
sns.heatmap(corr, annot=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X = data.drop('Outcome', axis=1)
y = data['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=1)