import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pima_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima_df.head(5)
pima_df.shape
pima_df.info()
pima_df.describe().transpose()
pima_df[~pima_df.applymap(np.isreal).all(1)]
pima_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = pima_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
pima_df.isnull().any()
pima_df.isna().sum()
pima_df['Glucose'].fillna(pima_df['Glucose'].mean(), inplace=True)
pima_df['BloodPressure'].fillna(pima_df['BloodPressure'].mean(), inplace=True)
pima_df['SkinThickness'].fillna(pima_df['SkinThickness'].median(), inplace=True)
pima_df['Insulin'].fillna(pima_df['Insulin'].median(), inplace=True)
pima_df['BMI'].fillna(pima_df['BMI'].median(), inplace=True)
pima_df['Outcome'] = pima_df['Outcome'].astype('category')
pass
pima_df['Outcome'].value_counts(normalize=True)
pima_df.groupby(['Outcome']).mean()
pass
from scipy.stats import zscore
numeric_cols = pima_df.drop('Outcome', axis=1)
class_values = pd.DataFrame(pima_df[['Outcome']])
numeric_cols = numeric_cols.apply(zscore)
pima_df_z = numeric_cols.join(class_values)
pima_df_z.head()
corr = pima_df[pima_df.columns].corr()
pass
import matplotlib.pylab as plt
pima_df_z.boxplot(by='Outcome', layout=(3, 4), figsize=(15, 20))
pima_df_z.hist('Age')
pima_df_z['log_age'] = np.log(pima_df_z['Age'])
pima_df_z['log_test'] = np.log(pima_df_z['Insulin'])
pima_df_z['log_preg'] = np.log(pima_df_z['Pregnancies'])
pima_df_z.hist('log_age')
pima_df_z.hist('log_test')
pima_df_z.hist('log_preg')
pass
pima_df.describe().transpose()
import scipy.stats as st
Mu = 72.4
Std = 12.09
sample_avg_bp = np.average(pima_df['BloodPressure'])
std_error_bp = Std / np.sqrt(pima_df.size)
print('Sample Avg BP : ', sample_avg_bp)
print('Standard Error: ', std_error_bp)
Z_norm_deviate = (sample_avg_bp - Mu) / std_error_bp
print('Normal Deviate Z value :', Z_norm_deviate)
p_value = st.norm.sf(abs(Z_norm_deviate)) * 2
print('p values', p_value)
if p_value > 0.05:
    print('Samples are likely drawn from the same distributions (fail to reject H0)')
else:
    print('Samples are likely drawn from different distributions (reject H0)')
Mu = 72.4
x = pima_df['BloodPressure']
est_pop_std = np.sqrt(np.sum(abs(x - x.mean()) ** 2) / (pima_df.size - 1))
sample_avg_bp = pima_df['BloodPressure'].mean()
std_error_bp = est_pop_std / np.sqrt(pima_df.size)
T_Statistic = (sample_avg_bp - Mu) / std_error_bp
pvalue = st.t.sf(np.abs(T_Statistic), pima_df.size - 1) * 2
print('Estimated Pop Stand Dev', est_pop_std)
print('Sample Avg BP : ', sample_avg_bp)
print('Standard Error: ', std_error_bp)
print('T Statistic', T_Statistic)
print('Pval', pvalue)
if pvalue > 0.05:
    print('Samples are likely drawn from the same distributions (fail to reject H0)')
else:
    print('Samples are likely drawn from different distributions (reject H0)')
pima_df_mod = pima_df.copy()
pima_df_mod['BloodPressure'] = pima_df_mod['BloodPressure'].mask(pima_df['BloodPressure'] == 0, pima_df['BloodPressure'].median())
from scipy.stats import ttest_ind
(stat, pvalue) = ttest_ind(pima_df_mod['BloodPressure'], pima_df['BloodPressure'])
print('compare means', pima_df_mod['BloodPressure'].mean(), pima_df['BloodPressure'].mean())
print('Tstatistic , Pvalue', stat, pvalue)
if pvalue > 0.05:
    print('Samples are likely drawn from the same distributions (fail to reject H0)')
else:
    print('Samples are likely drawn from different distributions (reject H0)')