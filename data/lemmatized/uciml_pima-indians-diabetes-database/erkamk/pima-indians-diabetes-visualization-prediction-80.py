import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.info()
data.describe()
data.columns
data.isnull().sum()
print(data.Outcome.value_counts())
labels = ('0', '1')
sizes = [500, 268]
colors = ['palegoldenrod', 'lightgrey']
explode = (0, 0)
pass
ax1.pie(sizes, colors=colors, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')
pass
data_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for each in data_columns:
    pass
    pass
    pass
    pass
    pass
df = pd.DataFrame(data, columns=data_columns)
pass
corrMatrix = df.corr()
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
g.despine(left=True)
g.set_axis_labels('Outcome', 'Glucose')
pass
pass
g.despine(left=True)
g.set_axis_labels('Outcome', 'Insluin')
pass
pass
pass
pass
pass
y = data.Outcome
data.drop(['Outcome'], axis=1, inplace=True)
x = (data - np.min(data)) / (np.max(data) - np.min(data))
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC
svm1 = SVC(gamma=0.01, C=10, kernel='rbf')