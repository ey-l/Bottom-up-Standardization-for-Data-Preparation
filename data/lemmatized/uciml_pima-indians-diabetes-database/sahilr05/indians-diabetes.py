import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.isnull().sum()
data.describe()
for i in data.columns:
    pass
    pass
from scipy import stats
z = np.abs(stats.zscore(data['Glucose']))
print(np.where(z > 3))
data = data[data['Glucose'] > 100]
data = data[data['BloodPressure'] > 60]
data = data[data['SkinThickness'] < 80]
data = data[data['BMI'] > 15]
data['Insulin'] = data['Insulin'].apply(lambda x: data['Insulin'].mean() if x == 0 else x)
data.describe()
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
y.value_counts()
from imblearn.over_sampling import SMOTE
sm = SMOTE()
(x, y) = sm.fit_resample(x, y)
data.corr()['Outcome']
pass
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, r2_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
(x_train, x_test, y_train, y_test) = train_test_split(x, y, stratify=y)
scle = StandardScaler()
x_train = scle.fit_transform(x_train)
x_test = scle.fit_transform(x_test)

def scoreChek(model):
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred)[0].apply(lambda x: 0 if x <= 0.5 else 1)
    print('Train Score', model.score(x_train, y_train))
    print('Test Score', model.score(x_test, y_test))
    print(classification_report(y_test, y_pred))
model = RandomForestClassifier()