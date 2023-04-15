import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.info()
data.columns
data.describe()
round(data.isnull().sum() / len(data) * 100, 2)
bmi_labels = ['Underweight', 'Healthy Weight', 'Overweight', 'Obesity']
data['BMI_Cat'] = pd.cut(data['BMI'], [0, 18.5, 25, 30, data['BMI'].max()], labels=bmi_labels)
data.head()
data['Glucose_Cat'] = data['Glucose'].apply(lambda x: 'Normal' if x < 140 else 'Prediabetes' if 140 <= x <= 199 else 'Risk')
data['Insulin'] = data['Insulin'].fillna(data.groupby(['BMI_Cat', 'Outcome', 'Glucose_Cat'])['Insulin'].transform('median'))
data['Insulin_Cat'] = data['Insulin'].apply(lambda x: 'Normal' if 16 <= x <= 166 else 'Abnormal')
data.info()
cat_var = ['BMI_Cat', 'Glucose_Cat', 'Insulin_Cat']
num_var = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
plt.figure(figsize=(15, 15))
sns.set(style='darkgrid')
i = 1
for col in cat_var:
    plt.subplot(4, 2, i)
    sns.countplot(data=data, x=col)
    i = i + 1

sns.pairplot(data, vars=num_var)

cor = data.corr()
plt.figure(figsize=(15, 5))
sns.heatmap(cor, annot=True)
