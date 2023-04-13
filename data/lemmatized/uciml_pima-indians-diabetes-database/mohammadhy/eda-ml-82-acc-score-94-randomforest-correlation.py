import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
pass
ax1.boxplot(df['Pregnancies'])
ax1.set_title('Pregnancies Outlier')
ax2.boxplot(df['Glucose'])
ax2.set_title('Glucose Outlier')
ax3.boxplot(df['BloodPressure'])
ax3.set_title('BloodPressure Outlier')
ax4.boxplot(df['SkinThickness'])
ax4.set_title('SkinThickness Outlier')
ax5.boxplot(df['Insulin'])
ax5.set_title('Insulin Outlier')
ax6.boxplot(df['BMI'])
ax6.set_title('BMI Outlier')
ax7.boxplot(df['DiabetesPedigreeFunction'])
ax7.set_title('DiabetesPedigreeFunction Outlier')
ax8.boxplot(df['Age'])
ax8.set_title('Age Outlier')
pass

def detect_outlier(data):
    q1 = np.quantile(data, 0.25)
    print('0.25: ', q1)
    q3 = np.quantile(data, 0.75)
    print('0.75:', q3)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    print('bottom:', low)
    high = q3 + 1.5 * iqr
    print('ceiling:', high)
    return (high, low)
(high, bottom) = detect_outlier(df['Pregnancies'])
df = df[(df['Pregnancies'] > bottom) & (df['Pregnancies'] < high)]
df['Pregnancies'].plot(kind='box')
pass
df['Pregnancies'].plot(kind='hist', bins=50)
(high, bottom) = detect_outlier(df['Age'])
df = df[(df['Age'] > bottom) & (df['Age'] < high)]
df['Age'].plot(kind='box')
pass
df['Age'].plot(kind='hist', bins=50)
(high, bottom) = detect_outlier(df['Glucose'])
df = df[(df['Glucose'] > bottom) & (df['Glucose'] < high)]
df['Glucose'].plot(kind='box')
pass
df['Glucose'].plot(kind='hist', bins=50)
(high, bottom) = detect_outlier(df['BloodPressure'])
df = df[(df['BloodPressure'] > bottom) & (df['BloodPressure'] < high)]
df['BloodPressure'].plot(kind='box')
pass
df['BloodPressure'].plot(kind='hist', bins=50)
df['Insulin'].value_counts()
print('Trustable_Mean:', stats.trim_mean(df['Insulin'], 0.2))
df['Insulin'] = df['Insulin'].replace(0, 53.94)
(high, bottom) = detect_outlier(df['Insulin'])
df = df[(df['Insulin'] > bottom) & (df['Insulin'] < high)]
df['Insulin'].plot(kind='box')
pass
df['Insulin'].plot(kind='hist', bins=100)
(high, bottom) = detect_outlier(df['BMI'])
df = df[(df['BMI'] > bottom) & (df['BMI'] < high)]
df['BMI'].plot(kind='box')
pass
df['BMI'].plot(kind='hist', bins=100)
(high, bottom) = detect_outlier(df['DiabetesPedigreeFunction'])
df = df[(df['DiabetesPedigreeFunction'] > bottom) & (df['DiabetesPedigreeFunction'] < high)]
df['DiabetesPedigreeFunction'].plot(kind='box')
pass
df['DiabetesPedigreeFunction'].plot(kind='hist', bins=100)
pass
ax1.boxplot(df['Pregnancies'])
ax1.set_title('Pregnancies Outlier')
ax2.boxplot(df['Glucose'])
ax2.set_title('Glucose Outlier')
ax3.boxplot(df['BloodPressure'])
ax3.set_title('BloodPressure Outlier')
ax4.boxplot(df['SkinThickness'])
ax4.set_title('SkinThickness Outlier')
ax5.boxplot(df['Insulin'])
ax5.set_title('Insulin Outlier')
ax6.boxplot(df['BMI'])
ax6.set_title('BMI Outlier')
ax7.boxplot(df['DiabetesPedigreeFunction'])
ax7.set_title('DiabetesPedigreeFunction Outlier')
ax8.boxplot(df['Age'])
ax8.set_title('Age Outlier')
pass
pass
print('method :pearson, kendal, spearman')
pass
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.preprocessing import MinMaxScaler, StandardScaler
min_max = MinMaxScaler()
stn = StandardScaler()
x_min_max = min_max.fit_transform(x)
x_finall = stn.fit_transform(x_min_max)
from sklearn.pipeline import Pipeline
pipe = Pipeline([('standard', StandardScaler()), ('min_max', MinMaxScaler())])
x1_finall = pipe.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x_finall, y)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
rn = RandomForestClassifier(n_estimators=100, max_depth=7)