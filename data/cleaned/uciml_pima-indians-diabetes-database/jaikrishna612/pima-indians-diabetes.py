import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.tail()
data.info()
data.isnull()
data.isnull().sum()
data.columns
data.describe()
data['Outcome'].unique()
data['Outcome'].value_counts()
sns.countplot(data['Outcome'])
data.corr()
sns.pairplot(data, diag_kind='hist')
sns.boxplot(data['Pregnancies'])
sns.boxplot(data['Glucose'])
sns.boxplot(data['BloodPressure'])
sns.boxplot(data['SkinThickness'])
sns.boxplot(data['Insulin'])
sns.boxplot(data['BMI'])
sns.boxplot(data['DiabetesPedigreeFunction'])
sns.boxplot(data['Age'])
per25 = data['Insulin'].quantile(0.25)
per75 = data['Insulin'].quantile(0.75)
iqr = per75 - per25
upper_limit = per75 + 1.5 * iqr
lower_limit = per25 - 1.5 * iqr
data[data['Insulin'] > upper_limit]
x_inp = data.iloc[:, :-1]
y_out = data.iloc[:, -1]
x_inp
y_out
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
(x_train, x_test, y_train, y_test) = train_test_split(x_inp, y_out, test_size=0.2, random_state=1)
x_test.shape
x_train.shape
lg = LogisticRegression(max_iter=200, solver='saga')