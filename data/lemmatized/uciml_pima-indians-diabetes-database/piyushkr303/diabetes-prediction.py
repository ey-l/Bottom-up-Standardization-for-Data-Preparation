import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat
import warnings
warnings.simplefilter('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.shape
df.columns
df.sample(10)
df.info()
print(df.isnull().mean() * 100)
duplicate_rows = df[df.duplicated()]
print('No. of duplicate rows: ', duplicate_rows.shape[0])
pass
pivot = pd.pivot_table(df, values='Outcome', columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], aggfunc='mean')
df.hist(figsize=(15, 8))
import scipy.stats as stats

def diagnostic_plot(data, col):
    pass
    fig.subplots_adjust(right=1.5)
    pass
    pass
    pass
    pass
    stats.probplot(data[col], dist='norm', fit=True, plot=plt)
    pass
    pass
    pass
    pass
for col in df:
    diagnostic_plot(df, col)
for col in df:
    print('Column name - ', col)
    print('Mean: {:.2f}'.format(df[col].mean()))
    print('Std: {:.2f}'.format(df[col].std()))
features = [feature for feature in df.columns if feature != 'diagnosis']
Q1 = df[features].quantile(0.25)
Q3 = df[features].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
df = df[~((df < Q1 - 1.5 * IQR) | (df > Q3 + 1.5 * IQR)).any(axis=1)]
print('No. of rows remaining: ', df.shape[0])
pass
corr = df.corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
pass
cormat.set_title('Correlation Matrix')
pass
import matplotlib
matplotlib.rcParams['figure.figsize'] = (30, 6)
plot_feature = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
for feature in plot_feature:
    pass
pass
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled
x = df.iloc[:, :-1]
x
y = df.iloc[:, -1:]
y
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
print('X_train:', x_train.shape)
print('X_test:', x_test.shape)
print('Y_train:', y_train.shape)
print('Y_test:', y_test.shape)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
modelLogistic = LogisticRegression()