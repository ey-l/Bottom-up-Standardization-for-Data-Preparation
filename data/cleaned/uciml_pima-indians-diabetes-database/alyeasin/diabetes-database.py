import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import missingno as msno
import warnings
warnings.simplefilter('ignore')
plt.style.use('dark_background')

df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(10)
df.shape
df.info()
df.describe().T.style.bar(subset=['mean'], color='#205fA2').background_gradient(subset=['std'], cmap='Reds').background_gradient(subset=['50%'], cmap='cividis')
feat = df.columns
col = (df[feat] == 0).sum()
print(col)
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df.isnull().sum()
msno.matrix(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']], figsize=(12, 8))
plt.grid()
df['Glucose'].fillna(df['Glucose'].median(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)
Glucose_Age_Insulin = df.groupby(['Glucose'])

def fillna_insulin(series):
    return series.fillna(series.median())
df['Insulin'] = Glucose_Age_Insulin['Insulin'].transform(fillna_insulin)
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df.isnull().sum()
df.isnull().values.any()
from matplotlib.pyplot import figure, show
plt.style.use('ggplot')
figure(figsize=(7, 4))
ax = sb.countplot(x=df['Outcome'], data=df, palette='husl', edgecolor='black', lw=3)
ax.set_xticklabels(['Healthy', 'Diabetic'])
(healthy, diabetics) = df['Outcome'].value_counts().values
print('Number of healthy people = ', healthy)
print('Number of diabetic people = ', diabetics)
plt.style.use('dark_background')
labels = ['Healthy', 'Diabetic']
df['Outcome'].value_counts().plot(kind='pie', labels=labels, subplots=True, autopct='%10.0f%%', labeldistance=2, figsize=(3, 3))
plt.style.use('dark_background')
sb.pairplot(df, hue='Outcome', palette='dark')
plt.style.use('dark_background')
plt.figure(dpi=90, figsize=(4, 4))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sb.heatmap(df.corr(), mask=mask, fmt='.1f', annot=True, lw=1, cmap='BuGn')
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title('Correlation Heatmap')

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
(xtrain, xtest, ytrain, ytest) = train_test_split(x, y, train_size=0.8, random_state=42)
print('xtrain data : ', xtrain.shape)
print('ytrain data : ', ytrain.shape)
print('xtest data : ', xtest.shape)
print('ytest data : ', ytest.shape)
from sklearn.linear_model import LinearRegression
model = LinearRegression()