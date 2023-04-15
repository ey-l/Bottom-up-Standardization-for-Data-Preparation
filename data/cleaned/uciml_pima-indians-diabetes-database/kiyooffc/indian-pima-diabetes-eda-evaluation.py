import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
diabetes_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_df
diabetes_df.info()
diabetes_df.describe()
diabetes_df.isna().sum()
import seaborn as sns
corr = diabetes_df.corr()
(_, ax) = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, ax=ax, cmap='twilight').set_title('Correlation Between Columns')
(_, ax) = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=diabetes_df['SkinThickness'], y=diabetes_df['Insulin'], hue=diabetes_df['Outcome'], ax=ax).set_title('Insulin vs SkinTickness')
(_, ax) = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=diabetes_df['Age'], y=diabetes_df['BMI'], hue=diabetes_df['Outcome'], ax=ax).set_title('Age vs BMI')
(_, ax) = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=diabetes_df['Glucose'], y=diabetes_df['Insulin'], hue=diabetes_df['Outcome'], ax=ax).set_title('Glucose vs Insulin')
(_, ax) = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=diabetes_df['BMI'], y=diabetes_df['BloodPressure'], hue=diabetes_df['Outcome'], ax=ax).set_title('BMI vs BloodPressure')
pd.plotting.scatter_matrix(diabetes_df, alpha=0.2, figsize=(15, 10))
diabetes_df['Outcome'].value_counts().plot(kind='bar')
diabetes_df['Insulin'][diabetes_df['Insulin'] == 0].value_counts()
for i in diabetes_df.columns:
    if i != 'Outcome':
        print(diabetes_df[i][diabetes_df[i] == 0].value_counts(), '\n----------------------\n')
X = diabetes_df.drop(columns=['Outcome'])
y = diabetes_df['Outcome']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()