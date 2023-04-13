import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pass
from sklearn.metrics import classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isna().sum()
df.columns
col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for i in col:
    df[i].replace(0, np.nan, inplace=True)
df.isna().sum()

def det_median(col_name):
    select_not_null = df[df[col_name].notnull()]
    result = select_not_null[[col_name, 'Outcome']].groupby(['Outcome'])[[col_name]].median().reset_index()
    return result
det_median('BMI')
det_median('Insulin')
det_median('Pregnancies')
det_median('BloodPressure')
det_median('SkinThickness')
det_median('DiabetesPedigreeFunction')
det_median('Glucose')
df.loc[(df['Outcome'] == 0) & df['BMI'].isnull(), 'BMI'] = 30.1
df.loc[(df['Outcome'] == 1) & df['BMI'].isnull(), 'BMI'] = 30.1
df.loc[(df['Outcome'] == 0) & df['Glucose'].isnull(), 'Glucose'] = 107.0
df.loc[(df['Outcome'] == 1) & df['Glucose'].isnull(), 'Glucose'] = 140.0
df.loc[(df['Outcome'] == 0) & df['BloodPressure'].isnull(), 'BloodPressure'] = 70.0
df.loc[(df['Outcome'] == 1) & df['BloodPressure'].isnull(), 'BloodPressure'] = 74.5
df.loc[(df['Outcome'] == 0) & df['Insulin'].isnull(), 'Insulin'] = 102.5
df.loc[(df['Outcome'] == 1) & df['Insulin'].isnull(), 'Insulin'] = 169.5
df.loc[(df['Outcome'] == 0) & df['SkinThickness'].isnull(), 'SkinThickness'] = 27.0
df.loc[(df['Outcome'] == 1) & df['SkinThickness'].isnull(), 'SkinThickness'] = 32.0
df.isna().sum()
pass
det_median('Age')
df.loc[(df['Outcome'] == 0) & (df['Age'] > 63), 'Age'] = 27
df.loc[(df['Outcome'] == 1) & (df['Age'] > 63), 'Age'] = 36
pass
pass
pass
pass
pass
pass
pass
pass
pass
df.loc[(df['Outcome'] == 0) & (df['BMI'] > 52), 'BMI'] = 30.1
df.loc[(df['Outcome'] == 1) & (df['BMI'] > 52), 'BMI'] = 30.1
df.loc[(df['Outcome'] == 0) & (df['BloodPressure'] > 105), 'BloodPressure'] = 70.0
df.loc[(df['Outcome'] == 1) & (df['BloodPressure'] > 105), 'BloodPressure'] = 74.5
df.loc[(df['Outcome'] == 0) & (df['Pregnancies'] > 12), 'Pregnancies'] = 2
df.loc[(df['Outcome'] == 1) & (df['Pregnancies'] > 12), 'Pregnancies'] = 4
df.loc[(df['Outcome'] == 0) & (df['Insulin'] > 250), 'Insulin'] = 102.5
df.loc[(df['Outcome'] == 1) & (df['Insulin'] > 250), 'Insulin'] = 169.5
df.loc[(df['Outcome'] == 0) & (df['SkinThickness'] > 40), 'SkinThickness'] = 27.0
df.loc[(df['Outcome'] == 1) & (df['SkinThickness'] > 40), 'SkinThickness'] = 32.0
df.loc[(df['Outcome'] == 0) & (df['DiabetesPedigreeFunction'] > 1), 'DiabetesPedigreeFunction'] = 0.336
df.loc[(df['Outcome'] == 1) & (df['DiabetesPedigreeFunction'] > 1), 'DiabetesPedigreeFunction'] = 0.449
df.loc[(df['Outcome'] == 0) & (df['SkinThickness'] < 20), 'SkinThickness'] = 27.0
df.loc[(df['Outcome'] == 1) & (df['SkinThickness'] < 20), 'SkinThickness'] = 32.0
df.loc[(df['Outcome'] == 0) & (df['BloodPressure'] < 40), 'BloodPressure'] = 70.0
df.loc[(df['Outcome'] == 1) & (df['BloodPressure'] < 40), 'BloodPressure'] = 74.5
pass
pass
pass
pass
pass
pass
pass
pass
pass
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()