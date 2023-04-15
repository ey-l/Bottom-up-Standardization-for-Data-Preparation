import pandas as pd
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.groupby('Outcome')['Outcome'].agg('count')
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
(class_count_0, class_count_1) = data['Outcome'].value_counts()
class_0 = data[data['Outcome'] == 0]
class_1 = data[data['Outcome'] == 1]
class_1_over = class_1.sample(class_count_0, replace=True)
data2 = pd.concat([class_1_over, class_0], axis=0)
data2.groupby('Outcome')['Outcome'].agg('count')
X = data2.drop(['Outcome'], axis=1)
y = data2[['Outcome']]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=11)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
bag_model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, random_state=11)