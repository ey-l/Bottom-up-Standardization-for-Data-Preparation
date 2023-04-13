import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.tail()
data.info()
data.isnull().sum()
data.dtypes
data.describe()
x = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']
print(y.value_counts())
pass
pass
pass
pass
x.describe()
data_std = (x - x.mean()) / x.std()
data_std.describe()
data_part = pd.concat([y, data_std], axis=1)
data_part.head()
data_part = pd.melt(data_part, id_vars='Outcome', var_name='features', value_name='value')
data_part.head()
pass
pass
pass
pass
pass
pass
pass
pass
pass
preg = pd.concat([y, data_std['Pregnancies']], axis=1)
preg = pd.melt(preg, id_vars='Outcome', var_name='Pregnancies', value_name='values')
preg.head()
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
x['Glucose'].unique()
pass
pass
Glucose = pd.concat([y, data_std['Glucose']], axis=1)
Glucose = pd.melt(Glucose, id_vars='Outcome', var_name='Glucose', value_name='values')
Glucose.head()
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
x['BloodPressure'].unique()
pass
pass
Bp = pd.concat([y, data_std['BloodPressure']], axis=1)
Bp = pd.melt(Bp, id_vars='Outcome', var_name='Bp', value_name='values')
Bp.head()
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
St = pd.concat([y, data_std['SkinThickness']], axis=1)
St = pd.melt(St, id_vars='Outcome', var_name='SkinThickness', value_name='values')
St.head()
pass
pass
pass
pass
pass
pass
pass
pass
x['Insulin'].sort_values().value_counts()
pass
pass
insulin = pd.concat([y, data_std['Insulin']], axis=1)
insulin = pd.melt(insulin, id_vars='Outcome', var_name='insulin', value_name='value')
pass
pass
pass
pass
pass
x['BMI'].sort_values().value_counts()
pass
pass
insulin = pd.concat([y, data_std['BMI']], axis=1)
insulin = pd.melt(insulin, id_vars='Outcome', var_name='BMI', value_name='value')
pass
pass
pass
pass
x['DiabetesPedigreeFunction'].sort_values().value_counts()
pass
pass
DPF = pd.concat([y, data_std['DiabetesPedigreeFunction']], axis=1)
DPF = pd.melt(DPF, id_vars='Outcome', var_name='DPF', value_name='value')
pass
pass
pass
x['Age'].value_counts()
pass
pass
Age = pd.concat([y, data_std['Age']], axis=1)
Age = pd.melt(Age, id_vars='Outcome', var_name='Age', value_name='value')
Age.head()
pass
pass