import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe()
df[df.isna()].count()
df[df == 0].count().sort_values(ascending=False)
Insulin_by_outcome = df.groupby(['Outcome']).mean()['Insulin']

def fixInsulin(insulin, outcome):
    if insulin == 0:
        return Insulin_by_outcome[outcome]
    else:
        return insulin
df['Insulin'] = df.apply(lambda row: fixInsulin(row['Insulin'], row['Outcome']), axis=1)
skinThickness_by_outcome = df.groupby(['Outcome']).mean()['SkinThickness']

def fixSkinThickness(skinThickness, outcome):
    if skinThickness == 0:
        return skinThickness_by_outcome[outcome]
    else:
        return skinThickness
df['SkinThickness'] = df.apply(lambda row: fixSkinThickness(row['SkinThickness'], row['Outcome']), axis=1)
bloodPressure_by_outcome = df.groupby(['Outcome']).mean()['BloodPressure']

def fixBloodPressure(bloodPressure, outcome):
    if bloodPressure == 0:
        return bloodPressure_by_outcome[outcome]
    else:
        return bloodPressure
df['BloodPressure'] = df.apply(lambda row: fixBloodPressure(row['BloodPressure'], row['Outcome']), axis=1)
glucose_by_outcome = df.groupby(['Outcome']).mean()['Glucose']

def fixGlucose(glucose, outcome):
    if glucose == 0:
        return glucose_by_outcome[outcome]
    else:
        return glucose
df['Glucose'] = df.apply(lambda row: fixGlucose(row['Glucose'], row['Outcome']), axis=1)
bim_by_outcome = df.groupby(['Outcome']).mean()['BMI']

def fixBMI(bim, outcome):
    if bim == 0:
        return bim_by_outcome[outcome]
    else:
        return bim
df['BMI'] = df.apply(lambda row: fixBMI(row['BMI'], row['Outcome']), axis=1)

def getSex(pregnancy):
    if pregnancy > 0:
        return 1
    else:
        return 0
df['Sex'] = df.apply(lambda row: getSex(row['Pregnancies']), axis=1)
df = df[df.Glucose > 50]
print(len(df))
df = df[(df.BloodPressure > 42) | (df.BloodPressure < 116)]
print(len(df))
df = df[(df.SkinThickness > 5) | (df.SkinThickness < 58)]
print(len(df))
df = df[df.Insulin < 625]
print(len(df))
df = df[(df.BMI > 15) | (df.BMI < 55)]
print(len(df))
df = df[df.DiabetesPedigreeFunction < 55]
print(len(df))
df = df[df.Age < 70]
print(len(df))
BMI_OMSNutritional_map = {-1: (0, 18.5), 0: (18.5, 24.9), 1: (24.9, 29.9), 2: (29.9, 34.9), 3: (34.9, 39.9), 4: (39.9, 1000)}

def getBMIClass(bmi):
    bmi_class = -100
    for (limit_index, limit) in enumerate(BMI_OMSNutritional_map.values()):
        if int(bmi) >= limit[0] and int(bmi) < limit[1]:
            bmi_class = list(BMI_OMSNutritional_map.keys())[limit_index]
            break
    if bmi_class == -100:
        print('Assined -100 class for: %d' % bmi)
    return bmi_class
df['BMI_class'] = df.apply(lambda row: getBMIClass(row['BMI']), axis=1)
pass
OMS_Glucose_map = {0: (0, 140), 1: (140, 200), 2: (200, 1000)}

def getGlucoseClass(glucose):
    glucose_class = 'None'
    for (limit_index, limit) in enumerate(OMS_Glucose_map.values()):
        if glucose >= limit[0] and glucose < limit[1]:
            glucose_class = list(OMS_Glucose_map.keys())[limit_index]
            break
    return glucose_class
df['Glucose_Class'] = df.apply(lambda row: getGlucoseClass(row['Glucose']), axis=1)
pass
Pressure_map = {-1: (60, 90), 0: (90, 140), 1: (140, 1000)}

def getPressureClass(pressure):
    pressure_class = -2
    for (limit_index, limit) in enumerate(Pressure_map.values()):
        if pressure >= limit[0] and pressure < limit[1]:
            pressure_class = list(Pressure_map.keys())[limit_index]
            break
    return pressure_class
df['BloodPressure_Class'] = df.apply(lambda row: getPressureClass(row['BloodPressure']), axis=1)
pass

def getInsulinClass(insulin):
    if insulin >= 100 and insulin <= 126:
        return 0
    else:
        return 1
df['Insulin_Class'] = df.apply(lambda row: getInsulinClass(row['Insulin']), axis=1)
pass
df.corr()['Outcome'].sort_values(ascending=False)
df = pd.get_dummies(df)
X = df.drop(['Outcome'], axis=1)
Y = df.Outcome.values
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, stratify=df.Outcome, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
regressor = RandomForestClassifier(n_estimators=20)