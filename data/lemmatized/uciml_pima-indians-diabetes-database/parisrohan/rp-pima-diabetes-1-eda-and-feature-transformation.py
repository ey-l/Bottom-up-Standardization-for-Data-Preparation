import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import scipy.stats as stat
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
pass
import warnings
warnings.filterwarnings('ignore')
df_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df_data.head()
df_data.info()
df_data.shape
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass

def get_cols_with_missing_values(DataFrame):
    missing_na_columns = DataFrame.isnull().sum()
    return missing_na_columns[missing_na_columns > 0]
print(get_cols_with_missing_values(df_data))
df_data.describe()
print('Total 0 value outliers in Glucose: ', df_data[df_data.Glucose == 0].shape[0])
print('Total 0 value outliers in BloodPressure: ', df_data[df_data.BloodPressure == 0].shape[0])
print('Total 0 value outliers in Insulin: ', df_data[df_data.Insulin == 0].shape[0])
print('Total 0 value outliers in SkinThickness: ', df_data[df_data.SkinThickness == 0].shape[0])
print('Total 0 value outliers in BMI: ', df_data[df_data.BMI == 0].shape[0])
df_data2 = df_data[(df_data.BloodPressure != 0) & (df_data.Glucose != 0) & (df_data.BMI != 0)]
df_data2.drop(['SkinThickness'], axis=1, inplace=True)
df_data2.shape
df_data2.describe()

def plot_data(df, feature):
    pass
    pass
    df[feature].hist()
    pass
    stat.probplot(df[feature], dist='norm', plot=pylab)
plot_data(df_data2, 'Age')
df_data['Age_log'] = np.log(df_data['Age'])
plot_data(df_data, 'Age_log')
df_data['Age_reciprocal'] = 1 / df_data.Age
plot_data(df_data, 'Age_reciprocal')
df_data['Age_sqaure'] = df_data.Age ** (1 / 2)
plot_data(df_data, 'Age_sqaure')
df_data['Age_exponential'] = df_data.Age ** (1 / 1.2)
plot_data(df_data, 'Age_exponential')
(df_data['Age_Boxcox'], parameters) = stat.boxcox(df_data['Age'])
plot_data(df_data, 'Age_Boxcox')
df_data['Pregnancies_log'] = np.log(df_data['Pregnancies'] + 1)
plot_data(df_data, 'Pregnancies_log')