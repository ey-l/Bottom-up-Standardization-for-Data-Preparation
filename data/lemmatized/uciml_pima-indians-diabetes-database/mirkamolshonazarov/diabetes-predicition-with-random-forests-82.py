import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info()
zero_to_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for i in zero_to_nan:
    data[i] = data[i].replace(0, np.NaN)
    mean = int(data[i].mean(skipna=True))
    data[i] = data[i].replace(np.NaN, mean)
X = data.drop('Outcome', axis=1)
y = data['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0, test_size=0.2)
rfc = RandomForestClassifier(criterion='gini', max_depth=4, min_samples_leaf=1, min_samples_split=2)