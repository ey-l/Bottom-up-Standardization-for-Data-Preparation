import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.isnull().sum()
data.describe()
data.loc[(data.SkinThickness < 5) & (data.Outcome == 0), 'SkinThickness'] = int(data[data.Outcome == 0]['SkinThickness'].mean())
data.loc[(data.SkinThickness < 5) & (data.Outcome == 1), 'SkinThickness'] = int(data[data.Outcome == 1]['SkinThickness'].mean())
data.loc[(data.Insulin == 0) & (data.Outcome == 0), 'Insulin'] = int(data[data.Outcome == 0]['Insulin'].mean())
data.loc[(data.Insulin == 0) & (data.Outcome == 1), 'Insulin'] = int(data[data.Outcome == 1]['Insulin'].mean())
data.columns
sns.countplot(data=data, x='Outcome', hue='Outcome')
plt.title('Womans have diabetes')
(fig, ax) = plt.subplots(figsize=(5, 5))
sns.boxplot(y='Age', x='Outcome', hue='Outcome', data=data)
plt.title(' Age ')
for i in data.columns:
    plt.figsize = (12, 10)
    plt.hist(data[i])
    plt.title(i)

sns.lineplot(data=data, x='Age', hue='Age', y='Glucose')
sns.pairplot(x_vars=['Glucose', 'Pregnancies', 'BMI'], y_vars='Age', hue='Outcome', data=data)
sns.pairplot(data=data, hue='Outcome')
corr_matrix = data.corr()
corr_matrix
import seaborn as sns
corr_matrix = data.corr()
top_corr_features = corr_matrix.index
plt.figure(figsize=(8, 6))
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='RdYlGn')
X = np.array(data[['Pregnancies', 'BloodPressure', 'Glucose', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])
y = np.array(data.Outcome)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
tree_clf = DecisionTreeClassifier()
knn_clf = KNeighborsClassifier()
bgc_clf = BaggingClassifier()
gbc_clf = GradientBoostingClassifier()
abc_clf = AdaBoostClassifier()
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf), ('tree', tree_clf), ('knn', knn_clf), ('bg', bgc_clf), ('gbc', gbc_clf), ('abc', abc_clf)], voting='hard')