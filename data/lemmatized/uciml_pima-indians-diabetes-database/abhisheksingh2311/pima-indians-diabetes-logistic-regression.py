import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.describe().transpose()
pass
valcount = data['Outcome'].value_counts().values.tolist()
ax.set_xticklabels(['Diabetic' + ':' + str(valcount[0]), 'Non-Diabetic' + ':' + str(valcount[1])])
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
for col in data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]:
    print(col + ':' + str(data[col].isnull().sum()) + '\npercentage : ' + str(data[col].isnull().sum() / len(data[col]) * 100) + '\n')
pass
for (i, col) in enumerate(data.columns, start=0):
    if i < 8:
        pass
        pass
data = data.fillna(data.median())
data.isna().sum()
data.groupby(['Outcome']).count()
pass
array = data.values
x = array[:, 0:8]
y = array[:, 8]
test_size = 0.3
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=test_size, random_state=42)
model = LogisticRegression(solver='liblinear')