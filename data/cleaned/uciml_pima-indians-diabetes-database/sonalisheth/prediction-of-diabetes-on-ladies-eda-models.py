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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
sns.set()

data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
print(data.shape)
print('**-------------------------------------**\n')
print(data.info())
print('**-------------------------------------**\n')
print(data.describe())
print('**-------------------------------------**')
data.isnull().sum()
sns.heatmap(data.corr(), cbar=False, cmap='PuOr', annot=True)
data.describe()
col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for i in col:
    data[i].replace(0, data[i].mean(), inplace=True)
print('A histogram is a classic visualization tool that represents the distribution of one or more variables by counting the number of observations that fall within disrete bins.')
p = data.hist(figsize=(20, 20))
sns.scatterplot(x='Age', y='Insulin', data=data)
sns.boxplot(x='Outcome', y='Pregnancies', data=data)
data.var()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(data.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = data.Outcome
print(X)
print(y)
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(X, y, test_size=0.3, random_state=3)
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(C=1, penalty='l2')