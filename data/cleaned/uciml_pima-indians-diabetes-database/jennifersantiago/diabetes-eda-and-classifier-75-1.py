import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_data.head()
diabetes_data.dtypes
diabetes_data.describe()
print(len(diabetes_data.loc[diabetes_data.SkinThickness == 0]))
print(len(diabetes_data.loc[diabetes_data.Glucose == 0]))
print(len(diabetes_data.loc[diabetes_data.Insulin == 0]))
print(len(diabetes_data.loc[diabetes_data.BloodPressure == 0]))
print(len(diabetes_data.loc[diabetes_data.BMI == 0]))
print(len(diabetes_data.loc[(diabetes_data.Insulin == 0) & (diabetes_data.Outcome == 0)]))
unlikely_zeros = ['SkinThickness', 'Insulin']
diabetes_data[unlikely_zeros] = diabetes_data[unlikely_zeros].replace(0, np.nan)
diabetes_data = diabetes_data.loc[(diabetes_data.Glucose != 0) & (diabetes_data.BMI != 0) & (diabetes_data.BloodPressure != 0)]
diabetes_data.describe()
sns.histplot(x=diabetes_data.Outcome).set_title('Outcome Distribution')
sns.histplot(x=diabetes_data.Age).set_title('Age Distribution')
sns.swarmplot(x=diabetes_data.Outcome, y=diabetes_data.Age, s=1).set_title('Age and Outcome')
sns.histplot(x=diabetes_data.Glucose).set_title('Glucose Distribution')
sns.swarmplot(x=diabetes_data.Outcome, y=diabetes_data.Glucose, s=1).set_title('Glucose and Outcome')
sns.scatterplot(x=diabetes_data.Glucose, y=diabetes_data.Insulin, hue=diabetes_data.Outcome).set_title('Glucose, Insulin, and Outcome')
sns.swarmplot(x=diabetes_data.Outcome, y=diabetes_data.Insulin, s=1).set_title('Insulin Distribution')
sns.histplot(x=diabetes_data.DiabetesPedigreeFunction).set_title('Diabetes Pedigree Function Distribution')
sns.swarmplot(x=diabetes_data.Outcome, y=diabetes_data.DiabetesPedigreeFunction, s=1).set_title('Diabetes Pedigree Function and Outcome')
sns.swarmplot(x=diabetes_data.Outcome, y=diabetes_data.SkinThickness, s=1).set_title('Skin Thickness and Outcome')
sns.scatterplot(x=diabetes_data.BMI, y=diabetes_data.SkinThickness, hue=diabetes_data.Outcome).set_title('BMI, Skin Thickness, and Outcome')
sns.swarmplot(x=diabetes_data.Outcome, y=diabetes_data.BMI, s=1).set_title('BMI and Outcome')
sns.histplot(x=diabetes_data.BloodPressure).set_title('Blood Pressure Distribution')
sns.swarmplot(x=diabetes_data.Outcome, y=diabetes_data.BloodPressure, s=1).set_title('Blood Pressure and Outcome')
sns.histplot(x=diabetes_data.Pregnancies).set_title('Pregnancies Distribution')
sns.swarmplot(x=diabetes_data.Outcome, y=diabetes_data.Pregnancies, s=1).set_title('Pregnancies and Outcomes')
diabetes_data.corr()
diabetes_data['PedigreeOverTime'] = diabetes_data.DiabetesPedigreeFunction * diabetes_data.Age
diabetes_data.head()
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'BMI', 'Age', 'PedigreeOverTime']
X = diabetes_data[features]
y = diabetes_data.Outcome
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, random_state=1, stratify=y)
imputer = SimpleImputer(strategy='median')
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_train.columns = X_train.columns
imputed_X_valid = pd.DataFrame(imputer.fit_transform(X_valid))
imputed_X_valid.columns = X_valid.columns
model = GradientBoostingClassifier(random_state=1, n_estimators=50, max_depth=4)