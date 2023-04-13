import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info(verbose=True)
print(data.describe())
print(data.shape)
data.isnull().sum()
corr = data.corr()
pass
pass
data.corr()
pass
pass
pass
Outcome_true = len(data.loc[data['Outcome'] == 0])
Outcome_false = len(data.loc[data['Outcome'] == 1])
print(Outcome_true, Outcome_false)
from sklearn.model_selection import train_test_split
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=10)
print(X.shape, y.shape)
print('total number of data {}'.format(data.shape[0]))
print('total number of prenancies {}'.format(len(data.loc[data['BMI'] == 0])))
print('total number of Glucose {}'.format(len(data.loc[data['Glucose'] == 0])))
print('total number of BloodPressure {}'.format(len(data.loc[data['BloodPressure'] == 0])))
print('total number of SkinThickness {}'.format(len(data.loc[data['SkinThickness'] == 0])))
print('total number of Insulin {}'.format(len(data.loc[data['Insulin'] == 0])))
print('total number of BMI {}'.format(len(data.loc[data['BMI'] == 0])))
print('total number of DiabetesPedigreeFunction {}'.format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print('total number of Age {}'.format(len(data.loc[data['Age'] == 0])))
from sklearn.impute import SimpleImputer
fill_values = SimpleImputer(missing_values=0, strategy='mean', verbose=0)
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)