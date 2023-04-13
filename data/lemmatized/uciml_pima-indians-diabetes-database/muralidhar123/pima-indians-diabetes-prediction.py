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
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.plot()
df.isnull().sum()
pass
plotnumber = 1
for column in df:
    if plotnumber <= 9:
        pass
        pass
        pass
    plotnumber += 1
df.describe()
df['Pregnancies'] = df['Pregnancies'].replace(0, df['Pregnancies'].mean())
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())
df.describe()
pass
plotnumber = 1
for column in df:
    if plotnumber <= 9:
        pass
        pass
        pass
    plotnumber += 1
X = df.drop(columns='Outcome')
y = df['Outcome']
pd.crosstab(df.Age, df.BloodPressure).plot(kind='bar', figsize=(20, 6))
pass
pass
pass
pass
pass
q = df['Pregnancies'].quantile(0.98)
data_cleaned = df[df['Pregnancies'] < q]
q = df['BloodPressure'].quantile(0.97)
data_cleaned = df[df['BloodPressure'] < q]
q = df['SkinThickness'].quantile(0.97)
data_cleaned = df[df['SkinThickness'] < q]
q = df['Insulin'].quantile(0.94)
data_cleaned = df[df['Insulin'] < q]
q = df['BMI'].quantile(0.97)
data_cleaned = df[df['BMI'] < q]
q = df['DiabetesPedigreeFunction'].quantile(0.99)
data_cleaned = df[df['DiabetesPedigreeFunction'] < q]
q = df['Age'].quantile(0.98)
data_cleaned = df[df['Age'] < q]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif['Features'] = X.columns
vif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y, test_size=0.4, random_state=120)
rf = RandomForestClassifier()