import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
diab_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diab_df.head(5)
diab_df.duplicated().sum()
diab_df.shape
diab_df.isnull().sum()
diab_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diab_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
diab_df.isnull().sum()
diab_df.hist(figsize=(11, 11))
diab_df['Glucose'].fillna(diab_df['Glucose'].mean(), inplace=True)
diab_df['BloodPressure'].fillna(diab_df['BloodPressure'].mean(), inplace=True)
diab_df['SkinThickness'].fillna(diab_df['SkinThickness'].mean(), inplace=True)
diab_df['Insulin'].fillna(diab_df['Insulin'].mean(), inplace=True)
diab_df['BMI'].fillna(diab_df['BMI'].mean(), inplace=True)
diab_df.isnull().sum()
import matplotlib.pyplot as plt
labels = ['No Diabetes', 'Diabetes']
colormap = {'lightgrey', 'tab:orange'}
diab_df['Outcome'].value_counts().plot.pie(startangle=90, colors=colormap, labels=labels)
plt.title('Diabetes Overall Sample Proportion', fontsize=15)
plt.ylabel('')
circle = plt.Circle((0, 0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(circle)

diab_df['Outcome'].value_counts()
import seaborn as sns
(fig, axes) = plt.subplots(2, 4, figsize=(18, 12))
fig.suptitle('Diabetes Outcome Distribution WRT All Independent Variables', fontsize=16)
sns.boxplot(ax=axes[0, 0], x=diab_df['Outcome'], y=diab_df['Pregnancies'], hue=diab_df['Outcome'], palette=('#23C552', '#C52219'))
axes[0, 0].set_title('Diabetes Outcome vs Pregnancies', fontsize=12)
sns.boxplot(ax=axes[0, 1], x=diab_df['Outcome'], y=diab_df['Glucose'], hue=diab_df['Outcome'], palette=('#23C552', '#C52219'))
axes[0, 1].set_title('Diabetes Outcome vs Glucose', fontsize=12)
sns.boxplot(ax=axes[0, 2], x=diab_df['Outcome'], y=diab_df['BloodPressure'], hue=diab_df['Outcome'], palette=('#23C552', '#C52219'))
axes[0, 2].set_title('Diabetes Outcome vs BloodPressure', fontsize=12)
sns.boxplot(ax=axes[0, 3], x=diab_df['Outcome'], y=diab_df['SkinThickness'], hue=diab_df['Outcome'], palette=('#23C552', '#C52219'))
axes[0, 3].set_title('Diabetes Outcome vs SkinThickness', fontsize=12)
sns.boxplot(ax=axes[1, 0], x=diab_df['Outcome'], y=diab_df['Insulin'], hue=diab_df['Outcome'], palette=('#23C552', '#C52219'))
axes[1, 0].set_title('Diabetes Outcome vs Insulin', fontsize=12)
sns.boxplot(ax=axes[1, 1], x=diab_df['Outcome'], y=diab_df['BMI'], hue=diab_df['Outcome'], palette=('#23C552', '#C52219'))
axes[1, 1].set_title('Diabetes Outcome vs BMI', fontsize=12)
sns.boxplot(ax=axes[1, 2], x=diab_df['Outcome'], y=diab_df['DiabetesPedigreeFunction'], hue=diab_df['Outcome'], palette=('#23C552', '#C52219'))
axes[1, 2].set_title('Diabetes Outcome vs DiabetesPedigreeFunction', fontsize=12)
sns.boxplot(ax=axes[1, 3], x=diab_df['Outcome'], y=diab_df['Age'], hue=diab_df['Outcome'], palette=('#23C552', '#C52219'))
axes[1, 3].set_title('Diabetes Outcome vs Age', fontsize=12)
sns.pairplot(diab_df, hue='Outcome', palette=('#23C552', '#C52219'))
plt.figure(figsize=(12, 10))
sns.heatmap(diab_df.corr(), annot=True, cmap='RdYlGn')
plt.title('Feature Correlation Matrix', fontsize=20)

from sklearn.model_selection import train_test_split
x = diab_df.drop(['Outcome'], axis=1)
y = diab_df['Outcome']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scaled = sc.fit_transform(x)
(x_train, x_test, y_train, y_test) = train_test_split(x_scaled, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()