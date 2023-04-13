import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info()
data.describe(include='all')
pass
data.isnull().sum()
error = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
data[error].isin([0]).sum()
data[error] = data[error].replace(0, np.NaN)
data.isnull().sum()
from sklearn.impute import SimpleImputer
si = SimpleImputer(missing_values=np.NaN, strategy='mean')
data[error] = si.fit_transform(data[error])
data.isnull().sum()
pass
data['Outcome'].value_counts()
from sklearn.utils import resample
data_major = data[data['Outcome'] == 0]
data_minor = data[data['Outcome'] == 1]
upsample = resample(data_minor, replace=True, n_samples=500, random_state=42)
df = pd.concat([upsample, data_major])
pass
df['Outcome'].value_counts()
pass
corr = df.corr()
pass
X = df.drop('Outcome', axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=2529)
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()