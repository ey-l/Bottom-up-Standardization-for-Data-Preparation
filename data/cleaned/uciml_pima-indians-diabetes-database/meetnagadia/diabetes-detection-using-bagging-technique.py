import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.columns
data.shape
data.head()
data.tail()
data.isnull().sum()
sns.countplot(x=data.Outcome)
plt.title('Plotting the count of patient with Diabetes')

data.Outcome.value_counts()
(f, axes) = plt.subplots(2, 4, figsize=(15, 6))
sns.boxplot(x='Glucose', data=data, palette='PRGn', ax=axes[0, 0])
sns.boxplot(x='Pregnancies', data=data, palette='PRGn', ax=axes[0, 1])
sns.boxplot(x='BloodPressure', data=data, palette='PRGn', ax=axes[0, 2])
sns.boxplot(x='SkinThickness', data=data, palette='PRGn', ax=axes[0, 3])
sns.boxplot(x='Insulin', data=data, palette='PRGn', ax=axes[1, 0])
sns.boxplot(x='BMI', data=data, palette='PRGn', ax=axes[1, 1])
sns.boxplot(x='DiabetesPedigreeFunction', data=data, palette='PRGn', ax=axes[1, 2])
sns.boxplot(x='Age', data=data, palette='PRGn', ax=axes[1, 3])

sns.jointplot(data=data[['Age', 'BloodPressure', 'Glucose', 'BMI']], height=10, ratio=5, color='r')
plt.title('Plotting the jointplot')

sns.pairplot(data, hue='Outcome')

(f, ax) = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, ax=ax, cmap='flare')

X = data.drop('Outcome', axis=1)
y = data['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
kfold = model_selection.KFold(n_splits=3)
base_cls = DecisionTreeClassifier()
num_trees = 200
model = BaggingClassifier(base_estimator=base_cls, n_estimators=num_trees)