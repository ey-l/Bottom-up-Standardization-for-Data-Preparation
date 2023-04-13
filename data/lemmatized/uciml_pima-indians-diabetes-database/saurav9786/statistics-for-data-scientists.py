import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
p = 0.1
n = 100
k = np.arange(0, 101)
binomial = stats.binom.pmf(k, n, p)
binomial
pass
pass
pass
pass
rate = 6
n = np.arange(0, 20)
poisson = stats.poisson.pmf(n, rate)
poisson
poisson[4]
poisson[4] + poisson[5] + poisson[6]
z = (1000000 - 700000) / 90000
z
1 - stats.norm.cdf(3.333)
z1 = (600000 - 700000) / 90000
z2 = (900000 - 700000) / 90000
stats.norm.cdf(z2) - stats.norm.cdf(z1)
z = (400000 - 70000) / 90000
z
stats.norm.cdf(-0.333)
from scipy.stats import uniform
n = 10000
start = 10
width = 20
data_uniform = uniform.rvs(size=n, loc=start, scale=width)
pass
pass
from scipy.stats import gamma
data_gamma = gamma.rvs(a=5, size=10000)
pass
pass
from scipy.stats import expon
data_expon = expon.rvs(scale=1, loc=0, size=1000)
pass
pass
from scipy.stats import bernoulli
data_bern = bernoulli.rvs(size=10000, p=0.6)
pass
pass
from numpy import mean
example = np.linspace(0, 8)
result = mean(example)
result
from numpy import array
M = array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
print(M)
col_mean = mean(M, axis=0)
print(col_mean)
row_mean = mean(M, axis=1)
print(row_mean)
from numpy import var
v = array([1, 2, 3, 4, 5, 6])
print(v)
result = var(v, ddof=1)
print(result)
M = array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
print(M)
col_mean = var(M, ddof=1, axis=0)
print(col_mean)
row_mean = var(M, ddof=1, axis=1)
print(row_mean)
from numpy import array
from numpy import std
M = array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
print(M)
col_mean = std(M, ddof=1, axis=0)
print(col_mean)
row_mean = std(M, ddof=1, axis=1)
print(row_mean)
from numpy import cov
x = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x)
y = array([9, 8, 7, 6, 5, 4, 3, 2, 1])
print(y)
Sigma = cov(x, y)[0, 1]
print(Sigma)
from numpy import corrcoef
x = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x)
y = array([9, 8, 7, 6, 5, 4, 3, 2, 1])
print(y)
Sigma = corrcoef(x, y)
print(Sigma)
pima_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima_df.head()
pima_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = pima_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
pima_df.isnull().any()
pima_df.isna().sum()
pima_df.shape
pima_df['Glucose'].fillna(pima_df['Glucose'].mean(), inplace=True)
pima_df['BloodPressure'].fillna(pima_df['BloodPressure'].mean(), inplace=True)
pima_df['SkinThickness'].fillna(pima_df['SkinThickness'].median(), inplace=True)
pima_df['Insulin'].fillna(pima_df['Insulin'].median(), inplace=True)
pima_df['BMI'].fillna(pima_df['BMI'].median(), inplace=True)
series1 = pima_df.Insulin
series1.dtype

def central_limit_theorem(data, n_samples=1000, sample_size=500, min_value=0, max_value=768):
    b = {}
    for i in range(n_samples):
        x = np.unique(np.random.randint(min_value, max_value, size=sample_size))
        b[i] = data[x].mean()
    c = pd.DataFrame()
    c['sample'] = b.keys()
    c['Mean'] = b.values()
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
    pass
central_limit_theorem(series1, n_samples=5000, sample_size=500)
import scipy.stats as st
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
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon
seed(1)
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
(stat, p) = wilcoxon(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')
np.random.seed(1234)
data = np.random.randn(10) + 0.1
data1 = np.random.randn(10) * 5
data2 = data1 + data
stats.ttest_1samp(data, 0)
stats.ttest_rel(data2, data1)
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
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu
from numpy.random import rand
data1 = 50 + rand(100) * 20
data2 = 51 + rand(100) * 20
(stat, p) = mannwhitneyu(data1, data2)
print('Statistics = %.3f, p = %.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')
import numpy as np
from scipy import stats
np.random.seed(123)
maths = np.round(np.random.randn(10) * 10 + 90)
physics = np.round(np.random.randn(10) * 10 + 85)
(t, pVal) = stats.ttest_rel(maths, physics)
print('The probability that the two distributions are equal is {0:5.3f} .'.format(pVal))
import pandas as pd
import statsmodels.formula.api as sm
np.random.seed(123)
df = pd.DataFrame({'Maths': maths, 'Physics': physics})