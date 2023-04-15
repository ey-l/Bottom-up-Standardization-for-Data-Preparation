import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.info()
df.head()
df.describe().T
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df.head()
df.isnull().sum()
from sklearn.impute import KNNImputer
imputer = KNNImputer(missing_values=np.NaN)