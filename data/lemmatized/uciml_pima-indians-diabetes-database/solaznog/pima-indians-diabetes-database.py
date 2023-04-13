import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.describe(include='all')
df.info()
correlations = df.corr()
correlations
pass
pass
pass
pass
df.columns
df_subset = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']]
df_subset.head()
from sklearn.model_selection import train_test_split
(train, test) = train_test_split(df_subset, test_size=0.2, random_state=44, stratify=df['Outcome'])
print('Ones proportion in dataset: ', np.mean(df_subset.Outcome))
print('Ones proportion in test set: ', np.mean(test.Outcome))
print('Ones proportion in train set: ', np.mean(train.Outcome))
from sklearn.preprocessing import StandardScaler
X_train = np.asarray(train.drop('Outcome', axis=1))