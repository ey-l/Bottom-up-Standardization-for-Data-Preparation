import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import os
for (dirname, _, filenames) in os.walk('/input/pima-indians-diabetes-database/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
Pima_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
Pima_df.head()
Pima_df.shape
Pima_df.isnull().mean()
corr = Pima_df.corr()
corr['Outcome'].sort_values(ascending=False)
plt.figure(figsize=(20, 20))
top_corr_features = corr.index
g = sns.heatmap(Pima_df[top_corr_features].corr(), annot=True, cmap='RdYlGn')
Pima_df[['Glucose', 'BMI']] = Pima_df[['Glucose', 'BMI']].replace(0, np.NaN)
Pima_df.dropna(inplace=True)
X = Pima_df.iloc[:, 0:8]
y_actual = Pima_df.iloc[:, 8]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
mean = np.mean(X, axis=0)
print('Mean: (%d, %d)' % (mean[0], mean[1]))
standard_deviation = np.std(X, axis=0)
print('Standard deviation: (%d, %d)' % (standard_deviation[0], standard_deviation[1]))
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y_actual, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()