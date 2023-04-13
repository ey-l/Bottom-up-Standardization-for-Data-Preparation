import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
path = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
diabetes = pd.read_csv(path, na_values='0')
diabetes.head()
diabetes.info()
diabetes.isnull().sum()
persentage = diabetes.isnull().mean() * 100
persentage.apply('{:.2f}'.format)
diabetes.describe().applymap('{:,.2f}'.format)
import missingno as msno
msno.bar(diabetes)
msno.matrix(diabetes)
sorted_by_insulin = diabetes.sort_values('Insulin')
msno.matrix(sorted_by_insulin)
msno.heatmap(diabetes)
msno.dendrogram(diabetes)
from numpy.random import rand

def fill_dummy_values(df, scaling_factor=0.075):
    df_dummy = df.copy(deep=True)
    for col in df_dummy:
        col = df_dummy[col]
        col_null = col.isnull()
        num_nulls = col_null.sum()
        col_range = col.max() - col.min()
        dummy_values = rand(int(num_nulls))
        dummy_values = dummy_values - 2
        dummy_values = dummy_values * scaling_factor * col_range + col.min()
        col[col_null] = dummy_values
    return df_dummy
diabetes_dummy = fill_dummy_values(diabetes)
nullity = diabetes.Insulin.isnull() + diabetes.BMI.isnull()
diabetes_dummy.plot(x='Insulin', y='BMI', kind='scatter', alpha=0.5, c=nullity, cmap='rainbow', figsize=(20, 10))
list(diabetes.columns)
value_one = diabetes.Glucose.mean()
value_two = diabetes.Glucose.count() / diabetes.Glucose.sum()
print(value_one, value_one)
diabetes.dropna(subset=['Glucose'], how='any', inplace=True)
from sklearn.impute import SimpleImputer

def simple_imputation(df, method):
    df_imputed = df.copy(deep=True)
    imputer = SimpleImputer(strategy=str(method))
    df_imputed.iloc[:, :] = imputer.fit_transform(df_imputed)
    return df_imputed
diabetes_mean = simple_imputation(diabetes, 'mean')
diabetes_median = simple_imputation(diabetes, 'median')
diabetes_mode = simple_imputation(diabetes, 'most_frequent')
import matplotlib.pyplot as plt
pass
nullity = diabetes.Insulin.isnull() + diabetes.BMI.isnull()
imputations = {'Mean Imputation ': diabetes_mean, 'Median Imputation ': diabetes_median, 'Most Freq Imputation ': diabetes_mode}
for (ax, df_key) in zip(axes.flatten(), imputations):
    imputations[df_key].plot(x='Insulin', y='BMI', kind='scatter', alpha=0.8, c=nullity, cmap='rainbow', ax=ax, colorbar=False, title=df_key)
from fancyimpute import KNN
knn_imputer = KNN()
diabetes_knn = diabetes.copy(deep=True)
diabetes_knn.iloc[:, :] = knn_imputer.fit_transform(diabetes_knn)
from fancyimpute import IterativeImputer
MICE_imputer = IterativeImputer()
diabetes_MICE = diabetes.copy(deep=True)
diabetes_MICE.iloc[:, :] = MICE_imputer.fit_transform(diabetes_MICE)
import matplotlib.pyplot as plt
pass
nullity = diabetes.Insulin.isnull() + diabetes.BMI.isnull()
imputations = {'KNN Imputation ': diabetes_knn, 'MICE Imputation ': diabetes_MICE}
for (ax, df_key) in zip(axes.flatten(), imputations):
    imputations[df_key].plot(x='Insulin', y='BMI', kind='scatter', alpha=0.8, c=nullity, cmap='rainbow', ax=ax, colorbar=False, title=df_key)