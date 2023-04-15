import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.isnull().sum()
sns.countplot(x=df.Outcome)
(f, axes) = plt.subplots(2, 4, figsize=(15, 6))
sns.boxplot(x='Glucose', data=df, palette='PRGn', ax=axes[0, 0])
sns.boxplot(x='Pregnancies', data=df, palette='PRGn', ax=axes[0, 1])
sns.boxplot(x='BloodPressure', data=df, palette='PRGn', ax=axes[0, 2])
sns.boxplot(x='SkinThickness', data=df, palette='PRGn', ax=axes[0, 3])
sns.boxplot(x='Insulin', data=df, palette='PRGn', ax=axes[1, 0])
sns.boxplot(x='BMI', data=df, palette='PRGn', ax=axes[1, 1])
sns.boxplot(x='DiabetesPedigreeFunction', data=df, palette='PRGn', ax=axes[1, 2])
sns.boxplot(x='Age', data=df, palette='PRGn', ax=axes[1, 3])

sns.jointplot(data=df[['Age', 'BloodPressure', 'Glucose', 'BMI']], height=10, ratio=5, color='r')
sns.pairplot(df)

(f, ax) = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, linewidths=0.5, linecolor='black', fmt='.1f', ax=ax, cmap='gray_r')

X = df.drop('Outcome', axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
kfold = model_selection.KFold(n_splits=3)
base_cls = DecisionTreeClassifier()
num_trees = 400
model = BaggingClassifier(base_estimator=base_cls, n_estimators=num_trees)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print('accuracy :')
print(results.mean())