import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.head()
(fig, axs) = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
index = 0
axs = axs.flatten()
for (k, v) in dataset.iloc[:, 0:8].items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.plot()
dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
dataset.dropna(inplace=True)
X = dataset.iloc[:, 0:8].values
y = dataset[['Outcome']].values
from sklearn.preprocessing import StandardScaler