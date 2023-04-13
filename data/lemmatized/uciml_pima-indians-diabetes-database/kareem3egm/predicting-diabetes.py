import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_data.head()
diabetes_data.shape
diabetes_data.describe()
diabetes_data.info(verbose=True)
pass
diabetes_data_copy = diabetes_data.copy(deep=True)
diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(diabetes_data_copy.isnull().sum())
pass
plot = diabetes_data.hist(figsize=(20, 20))
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace=True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace=True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace=True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace=True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace=True)
diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(diabetes_data_copy.isnull().sum())
plot = diabetes_data_copy.hist(figsize=(20, 20))
pass
pass
pass
pass
from pandas_profiling import ProfileReport
profile = ProfileReport(diabetes_data.corr(), title='Pandas profiling report ', html={'style': {'full_width': True}})
profile.to_notebook_iframe()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = diabetes_data_copy.Outcome
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=1 / 3, random_state=42, stratify=y)
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
'\n#ensemble.VotingClassifier(estimators, voting=’hard’, weights=None,n_jobs=None, flatten_transform=None)\n'
LRModel_ = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=33)
RFModel_ = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5, random_state=33)
KNNModel_ = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto')
NNModel_ = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(1000, 20), learning_rate='constant', activation='relu', power_t=0.4, max_iter=250)
VotingClassifierModel = VotingClassifier(estimators=[('LRModel', LRModel_), ('RFModel', RFModel_), ('KNNModel', KNNModel_), ('NNModel', NNModel_)], voting='soft')