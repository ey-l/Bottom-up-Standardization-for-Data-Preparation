import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pass
import warnings
warnings.filterwarnings('ignore')
diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_data.head()
diabetes_data.isna().any(axis=0)
diabetes_data.info(verbose=True)
diabetes_data.describe()
diabetes_copy = diabetes_data.copy(deep=True)
diabetes_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(diabetes_copy.isnull().sum())
p = diabetes_data.hist(figsize=(20, 20))
diabetes_copy['Glucose'].fillna(diabetes_copy['Glucose'].mean(), inplace=True)
diabetes_copy['BloodPressure'].fillna(diabetes_copy['BloodPressure'].mean(), inplace=True)
diabetes_copy['SkinThickness'].fillna(diabetes_copy['SkinThickness'].mean(), inplace=True)
diabetes_copy['Insulin'].fillna(diabetes_copy['Insulin'].mean(), inplace=True)
diabetes_copy['BMI'].fillna(diabetes_copy['BMI'].mean(), inplace=True)
p = diabetes_copy.hist(figsize=(20, 20))
print(diabetes_data['Outcome'].value_counts())
p = diabetes_data['Outcome'].value_counts().plot(kind='bar')
diabetes_data.corr()
import seaborn as sns
from matplotlib import pyplot as plt
pass
pass
pass
pass
pass
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X = pd.DataFrame(X_sc.fit_transform(diabetes_copy.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPredigreeFunction', 'Age'])
X.head()
y = diabetes_copy.Outcome
y.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=1 / 3, random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
for i in range(1, 15):
    knn = KNeighborsClassifier(i)