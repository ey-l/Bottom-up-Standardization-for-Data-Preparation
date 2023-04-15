import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info()
data.dtypes
data.describe().T
data['Outcome'] = data['Outcome'].astype(str)
data['Outcome']
plt.hist(data['Outcome'])
plt.title('Outcome Distribution')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
data['Outcome'].value_counts(normalize=True).plot(kind='pie', legend=True, table=True, figsize=(10, 8))
data.hist(figsize=(20, 10))
data[data['SkinThickness'] > 80]
data = data[data['SkinThickness'] != 99]
data
data[data['Pregnancies'] > 15]
data = data[data['Pregnancies'] < 15]
data
missing_value = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
missing_value
data[missing_value]
data[missing_value] = np.where(data[missing_value] == 0, np.nan, data[missing_value])
data.info()
data.isnull().any()
data = data.reset_index(drop=True)
data.info()
data.hist(figsize=(20, 10))
data['Age'].describe()
data['age_bins'] = pd.cut(x=data['Age'], bins=[20, 30, 40, 50, 60, 70, 80, 90])
data.head()
data['age_bins'].dtype
data['age_bins'] = data['age_bins'].astype(str)
data.age_bins.value_counts()
data_half_clean = data.fillna(data.median())
data_half_clean
data_clean = data.fillna(data.groupby(['age_bins', 'Outcome', 'Pregnancies']).transform('median'))
data_clean
data_clean.isnull().any()
data.groupby(['age_bins', 'Outcome', 'Pregnancies']).head()
data[data.age_bins == '(60, 70]'].groupby(['age_bins', 'Outcome', 'Pregnancies']).head().sort_values(by=['Outcome', 'Pregnancies'])
data[(data.Outcome == '0') & (data.age_bins == '(20, 30]') & (data.Pregnancies == 0)]
data_clean[(data_clean.Outcome == '0') & (data_clean.age_bins == '(20, 30]') & (data_clean.Pregnancies == 0)]
data[(data.Outcome == '0') & (data.age_bins == '(60, 70]')]
data_clean = data_clean.fillna(data_clean.groupby(['age_bins', 'Outcome', 'Pregnancies']).transform('median'))
data_clean
data_clean.isnull().any()
data[(data.Outcome == '0') & (data.age_bins == '(60, 70]') & (data.Pregnancies == 1)]
data[(data.Outcome == '0') & (data.age_bins == '(70, 80]') & (data.Pregnancies == 2)]
data[(data.Outcome == '0') & (data.age_bins == '(20, 30]') & (data.Pregnancies == 0)]
data_clean[(data_clean.Outcome == '0') & (data_clean.age_bins == '(20, 30]') & (data_clean.Pregnancies == 0)]
data_clean = data_clean.fillna(data_clean.groupby(['age_bins', 'Outcome']).transform('median'))
data_clean.head()
data_clean.isnull().any()
data_clean = data_clean.fillna(data_clean.groupby(['Outcome']).transform('median'))
data[(data.Outcome == '0') & (data.age_bins == '(70, 80]')]
data_clean[(data_clean.Outcome == '0') & (data_clean.age_bins == '(70, 80]')]
data_clean.isnull().any()
data_clean.describe().T
data_half_clean.describe().T
data_corr = data_clean.copy()
data_corr
data_corr['Outcome'] = data_corr['Outcome'].astype(int)
data_corr.dtypes
data_corr.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(data_corr.corr(), cmap='rocket_r', annot=True)
data_clean
plt.figure(figsize=(10, 8))
colors = {'0': 'blue', '1': 'red'}
plt.scatter(data_half_clean.index, data_half_clean.Insulin, c=data_clean['Outcome'].map(colors))
plt.title("Each person's insulin values and diabetic status", fontsize=20)
plt.xlabel('Patient Index', fontsize=15)
plt.ylabel('Insulin', fontsize=15)
plt.figure(figsize=(10, 8))
colors = {'0': 'blue', '1': 'red'}
plt.scatter(data_clean.index, data_clean.Insulin, c=data_clean['Outcome'].map(colors))
plt.title("Each person's insulin values and diabetic status", fontsize=20)
plt.xlabel('Patient Index', fontsize=15)
plt.ylabel('Insulin', fontsize=15)
data_clean = data_clean.reset_index()
data_clean
data.head()
fig = px.scatter(data, x='Pregnancies', y='Insulin', color='Outcome', color_discrete_sequence=['red', 'blue'], width=800, height=400)
fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='LightSteelBlue')
fig.show()
data_insulin = data_clean.groupby('Outcome').agg({'Insulin': 'mean'}).reset_index()
px.bar(data_insulin, x='Outcome', y='Insulin', color='Outcome', title='Mean insulin value of diabetic and non-diabetic', width=400)
px.scatter(data_clean, x='BMI', y='Insulin', color='Outcome', color_discrete_sequence=['red', 'blue'])
px.scatter(data_clean, x='Glucose', y='Insulin', trendline='ols')
px.scatter(data_clean, x='index', y='Glucose', color='Outcome', color_discrete_sequence=['red', 'blue'], title="Each person's insulin values and diabetic status")
data_glucose = data_clean.groupby('Outcome').agg({'Glucose': 'mean'}).reset_index()
px.bar(data_glucose, x='Outcome', y='Glucose', color='Outcome', title='Mean glucose value of diabetic and non-diabetic', width=700)
px.scatter(data_clean, x='BMI', y='SkinThickness', trendline='ols', title='The relation between skin thickness and weight')