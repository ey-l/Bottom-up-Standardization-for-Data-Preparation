import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.isna().sum()
data.describe().T
corr = data.corr()
plt.figure(figsize=[10, 7])
sns.heatmap(corr, annot=True)
k = sns.countplot(data['Outcome'])
for b in k.patches:
    k.annotate(format(b.get_height(), '.0f'), (b.get_x() + b.get_width() / 2.0, b.get_height()))
sns.pairplot(data)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for feature in features:
    number = np.random.normal(data[feature].mean(), data[feature].std() / 2)
    data[feature].fillna(value=number, inplace=True)
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(data.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = data.Outcome
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=100)
lr = LogisticRegression()