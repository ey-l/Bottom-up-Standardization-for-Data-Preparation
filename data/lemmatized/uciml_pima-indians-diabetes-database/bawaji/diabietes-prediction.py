from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
BASE_DIR = Path('data/input/uciml_pima-indians-diabetes-database')
df = pd.read_csv(BASE_DIR / 'diabetes.csv')
df.head(10)
features_list = list(df.drop(columns='Outcome').columns)
columns = list(df.columns)
print(features_list)
isnull = df.isnull().sum()
isnull
dup = df.duplicated(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
dup.value_counts()
df.drop_duplicates(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

def standardize_var(x):
    mean = np.mean(x)
    std = np.sqrt(np.sum(np.square(x - mean)) / (len(x) - 1))
    return (x - mean) / std / np.sqrt(len(x) - 1)
sdf = df.apply(standardize_var)
sdf_X = sdf[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
corr = np.array(sdf_X.corr())
corr_inv = np.linalg.inv(corr)