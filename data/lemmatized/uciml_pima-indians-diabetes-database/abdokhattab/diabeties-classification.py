import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pass
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.sample(5)
df.info()
df.describe()
cols = ['BloodPressure', 'SkinThickness', 'BMI', 'Glucose']
df[cols] = df[cols].replace(0, np.NaN)
from sklearn.model_selection import train_test_split
(train, test) = train_test_split(df, test_size=0.2, random_state=42)
train.isnull().sum()
pass
for (i, col) in enumerate(train):
    pass
    pass
    pass
    pass
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100), max_iter=10, random_state=42)
imputed_train = imp.fit_transform(train)
train = pd.DataFrame(imputed_train, columns=train.columns)
train.isna().sum()
pass
for (i, col) in enumerate(train):
    pass
    pass
    pass
    pass

def detect_outliers(df):
    outliers = pd.DataFrame(columns=['Feature', 'Num of Outliers', 'Handled?'])
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        N_O_O = df.loc[(df[col] < fence_low) | (df[col] > fence_high)].shape[0]
        df.loc[df[col] < fence_low, col] = fence_low
        df.loc[df[col] > fence_high, col] = fence_high
        outliers = outliers.append({'Feature': col, 'Num of Outliers': N_O_O, 'Handled?': df[col].all() < fence_high}, ignore_index=True)
    return outliers
detect_outliers(train)
pass
for (i, col) in enumerate(train):
    pass
    pass
    pass
    pass
train.loc[train['Pregnancies'] > 13, 'Pregnancies'] = 13
train['Pregnancies'].value_counts()
test.isnull().sum()
imputed_test = imp.transform(test)
test = pd.DataFrame(imputed_test, columns=test.columns)
pass
for (i, col) in enumerate(test):
    pass
    pass
    pass
    pass
detect_outliers(test)
pass
pass
pass
pass
x_train = train.drop('Outcome', axis=1)
y_train = train['Outcome']
x_test = test.drop('Outcome', axis=1)
y_test = test['Outcome']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000)