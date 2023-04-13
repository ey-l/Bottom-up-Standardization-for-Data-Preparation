import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings('ignore')
diabete = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabete.head()
diabete.tail()
diabete.describe()
diabete.info()
diabete.isna().sum()
print('Dataset shape is', diabete.shape)
pass
diabe_corr = diabete.corr()
pass
feature_col = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for column in feature_col:
    diabete[column] = diabete[column].replace(0, np.NaN)
    mean = diabete[column].mean(skipna=True)
    diabete[column] = diabete[column].replace(np.NaN, mean)
x = diabete[feature_col]
y = diabete[['Outcome']]
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)
clf = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')