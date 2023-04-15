import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
db = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
db.head()
db.tail()
db.isnull().sum()
db.info()
db.describe()
diabetes_data_copy = db.copy(deep=True)
diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(diabetes_data_copy.isnull().sum())
p = diabetes_data_copy.hist(figsize=(20, 20))
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace=True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace=True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace=True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace=True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace=True)
p = diabetes_data_copy.hist(figsize=(20, 20))
print(diabetes_data_copy.isnull().sum())
diabetes_data_copy['Outcome'].value_counts()
diabetes_data_copy.shape
diabetes_data_copy.Outcome.unique()
plt.figure(figsize=(7, 5))
sns.countplot(x='Outcome', data=diabetes_data_copy, palette=('Orange', 'DarkBlue'))
plt.xlabel('Diabetes Disease (0 = No, 1= Yes)')

plt.figure(figsize=(10, 9))
sns.countplot(x='Age', hue='Outcome', data=diabetes_data_copy, palette=('Orange', 'DarkBlue'))
plt.xlabel('Age (0 = No, 1= Yes)')

plt.figure(figsize=(10, 9))
sns.countplot(x='Glucose', data=diabetes_data_copy, hue='Outcome', palette=('Orange', 'DarkBlue'))
plt.xlabel('The Slope of The Peak Glucose level')
plt.ylabel('Frequency of Diabetes Disease or Not')

plt.figure(figsize=(10, 9))
plt.subplot()
plt.title('subplot: 231')
sns.countplot(data=diabetes_data_copy, x='BloodPressure', hue='Outcome')

column_name = ['Pregnancies', 'SkinThickness', 'DiabetesPedigreeFunction', 'BMI', 'Age', 'BloodPressure']
diabetes_data_copy[column_name] = diabetes_data_copy[column_name].clip(lower=diabetes_data_copy[column_name].quantile(0.15), upper=diabetes_data_copy[column_name].quantile(0.85), axis=1)
diabetes_data_copy.plot(kind='box', figsize=(10, 8))
diabetes_data_copy.drop(columns=['Insulin'], axis=1, inplace=True)
diabetes_data_copy.plot(kind='box', figsize=(10, 8))
plt.figure(figsize=(10, 9))
sns.countplot(x='Pregnancies', hue='Outcome', data=diabetes_data_copy, palette=('Orange', 'DarkBlue'))

p = sns.pairplot(diabetes_data_copy, hue='Outcome', corner=True)
plt.figure(figsize=(12, 10))
p = sns.heatmap(diabetes_data_copy.corr(), annot=True, cmap='RdYlGn')
db1 = db
x = db1.drop(['Outcome'], axis=1)
y = db1['Outcome']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=0)
(x_train.shape, x_test.shape)
(y_train.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()