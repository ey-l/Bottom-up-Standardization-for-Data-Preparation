from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import warnings

sns.set()
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe().T
print(df['Outcome'].value_counts())
print(f"Approximately {df['Outcome'].value_counts()[0] / df['Outcome'].size * 100:.4f}% of patients don't have diabetes.")
print(f"Approximately {df['Outcome'].value_counts()[1] / df['Outcome'].size * 100:.4f}% of patients have diabetes.")
df.describe().T
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True)
(fig, axes) = plt.subplots(4, 2, figsize=(14, 14))
fig.tight_layout(pad=4.0)
for (i, j) in enumerate(df.columns[:-1]):
    sns.histplot(df[j], ax=axes[i // 2, i % 2])
ndf = df.copy(deep=True)
(ndf == 0).sum()
colsToFix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
ndf[colsToFix] = ndf[colsToFix].replace(0, np.NaN)
ndf.isnull().sum()

def gimmeThemStats(dFrame):
    """
    Description
    ----
    Outputs the general statistical description of the dataframe,
    outputs the correlation heatmap, and outputs a distribution plot.
    
    Parameters
    ----
    dFrame(DataFrame):
        The dataframe for which information will be displayed.
        
    Returns
    ----
    Nothing.
    
    """
    print('Descriptive Stats:')

    plt.figure(figsize=(10, 8))
    plt.title('Heatmap', fontsize='x-large')
    sns.heatmap(dFrame.corr(), annot=True)
    (fig, axes) = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle('Distribution Plot', y=0.92, fontsize='x-large')
    fig.tight_layout(pad=4.0)
    for (i, j) in enumerate(df.columns[:-1]):
        sns.distplot(dFrame[j], ax=axes[i // 2, i % 2])
gimmeThemStats(ndf)
dfMeanMed = ndf.copy(deep=True)
dfMeanMed[colsToFix].skew()
dfMeanMed['Glucose'].fillna(dfMeanMed['Glucose'].median(), inplace=True)
dfMeanMed['SkinThickness'].fillna(dfMeanMed['SkinThickness'].median(), inplace=True)
dfMeanMed['Insulin'].fillna(dfMeanMed['Insulin'].median(), inplace=True)
dfMeanMed['BMI'].fillna(dfMeanMed['BMI'].median(), inplace=True)
dfMeanMed['BloodPressure'].fillna(dfMeanMed['BloodPressure'].mean(), inplace=True)
dfMeanMed.isnull().sum()
gimmeThemStats(dfMeanMed)
dfMeanMed.kurt() - ndf.kurt()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
exEstimator = DecisionTreeRegressor(max_features='sqrt', random_state=42)
exStyle = 'descending'
exImputer = IterativeImputer(estimator=exEstimator, imputation_order=exStyle, random_state=42)
exImputer
edf = ndf.copy(deep=True)