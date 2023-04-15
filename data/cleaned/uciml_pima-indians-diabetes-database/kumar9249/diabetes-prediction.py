import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
filepath = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
data = pd.read_csv(filepath)
data.sample(10)
data.shape
data.info()
data.describe().T
data['Outcome'].value_counts()
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x='Outcome')

data.isnull().sum().any()
duplicate_rows = data[data.duplicated()]
duplicate_rows.shape[0]
data.hist(figsize=(12, 10))

plt.figure(figsize=(10, 8))
corr = data.corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
cormat = sns.heatmap(corr, mask=mask, linewidths=1, annot=True, fmt='.2f')
cormat.set_title('Correlation Matrix')

sns.set(style='ticks', color_codes=True)
sns.pairplot(data)


def diagnostic_plot(data, col):
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 3, 1)
    sns.distplot(data[col], bins=10)
    plt.title('Histogram')
    plt.subplot(1, 3, 2)
    stats.probplot(data[col], dist='norm', fit=True, plot=plt)
    plt.title('Q-Q Plot')
    plt.subplot(1, 3, 3)
    sns.boxplot(y=data[col])
    plt.title('Boxplot')

data['Pregnancies'].value_counts()
max_threshold = data['Pregnancies'].quantile(0.95)
data = data[data['Pregnancies'] <= max_threshold]
print('Maximum Age is: {}'.format(data['Age'].max()))
print('Minimum Age is: {}'.format(data['Age'].min()))
diagnostic_plot(data, 'Glucose')
data = data[data['Glucose'] >= 25]
diagnostic_plot(data, 'BloodPressure')
data = data[data['BloodPressure'] != 0]
diagnostic_plot(data, 'BMI')
data = data[(data['BMI'] > 10) & (data['BMI'] < 50)]
data['Insulin'].value_counts().sort_index(ascending=False)
data = data[data['Insulin'].between(15, 600)]
diagnostic_plot(data, 'SkinThickness')
data = data[data['SkinThickness'] < 60]
diagnostic_plot(data, 'DiabetesPedigreeFunction')
data.shape
X = data.drop(['Outcome'], axis=1)
y = data['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()