import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
diab_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diab_df.head(5)
diab_df.tail(5)
diab_df.duplicated().sum()
diab_df.shape
diab_df.info()
diab_df.isnull().sum()
diab_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diab_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
diab_df.isnull().sum()
diab_df.hist(figsize=(11, 11), color='#008080')
diab_df['Glucose'].fillna(diab_df['Glucose'].mean(), inplace=True)
diab_df['BloodPressure'].fillna(diab_df['BloodPressure'].mean(), inplace=True)
diab_df['SkinThickness'].fillna(diab_df['SkinThickness'].median(), inplace=True)
diab_df['Insulin'].fillna(diab_df['Insulin'].median(), inplace=True)
diab_df['BMI'].fillna(diab_df['BMI'].median(), inplace=True)
diab_df.isnull().sum()
diab_df.info()
pass
pass
pass
pass
pass
pass
pass
pass
fig.suptitle('Diabetes Outcome Distribution WRT All Independent Variables', fontsize=16)
pass
axes[0, 0].set_title('Diabetes Outcome vs Pregnancies', fontsize=12)
pass
axes[0, 1].set_title('Diabetes Outcome vs Glucose', fontsize=12)
pass
axes[0, 2].set_title('Diabetes Outcome vs BloodPressure', fontsize=12)
pass
axes[0, 3].set_title('Diabetes Outcome vs SkinThickness', fontsize=12)
pass
axes[1, 0].set_title('Diabetes Outcome vs Insulin', fontsize=12)
pass
axes[1, 1].set_title('Diabetes Outcome vs BMI', fontsize=12)
pass
axes[1, 2].set_title('Diabetes Outcome vs DiabetesPedigreeFunction', fontsize=12)
pass
axes[1, 3].set_title('Diabetes Outcome vs Age', fontsize=12)
pass
pass
pass
pass
diab_df.describe()
diab_df['Outcome'].value_counts()
from sklearn.model_selection import train_test_split
x = diab_df.drop(['Outcome'], axis=1)
y = diab_df['Outcome']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scaled = sc.fit_transform(x)
(x_train, x_test, y_train, y_test) = train_test_split(x_scaled, y, test_size=0.3, random_state=0)
(x_train.shape, y_train.shape)
(x_test.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()