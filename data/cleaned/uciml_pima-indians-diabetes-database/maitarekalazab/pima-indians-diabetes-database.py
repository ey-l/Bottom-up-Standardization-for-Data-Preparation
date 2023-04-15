import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.svm import SVC
dataset_path = 'data/input/uciml_pima-indians-diabetes-database/'
data = pd.read_csv(os.path.join(dataset_path, 'diabetes.csv'))
print('shape', data.shape)
data.head(5)
data.info()
data.describe()
data_null = data.isnull().sum()
print(data_null)
data.hist(bins=20, figsize=(15, 10))

sns.countplot(x='Outcome', data=data)
(f, ax) = plt.subplots(figsize=(10, 10))
corr = data.corr()
mp = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot=True)
mp.set_title(label='dataset correlation', fontsize=20)
sns.pairplot(data, hue='Outcome')
for col in data.columns:
    print('The minimum value for the {} column is {}'.format(col, data[col].min()))
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
data['Insulin'] = data['Insulin'].fillna(data['Insulin'].median())
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']:
    data[col] = data[col].fillna(data[col].mean())
(train, test) = train_test_split(data, test_size=0.3, random_state=50)
x_train = train.drop(columns='Outcome')
y_train = train['Outcome']
x_val = test.drop(columns='Outcome')
y_val = test['Outcome']