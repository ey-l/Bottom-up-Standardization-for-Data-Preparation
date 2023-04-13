import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import seaborn as sns
import matplotlib.pyplot as plt
pass
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.tail(5)
from PIL import Image
path = '../input/conclusion/conclusion.png'
df.describe()[1:6]
df.info()

def finding_zeros(frame):
    columns = frame.columns[:8]
    for i in columns:
        zeros = len(frame.loc[frame[i] == 0])
        print(f'The numbers of 0 values in {i} = {zeros}')
finding_zeros(df)
NAN_value = df.isnull().sum() / len(df) * 100
Missing = NAN_value[NAN_value == 0].index.sort_values(ascending=False)
Missing_data = pd.DataFrame({'Missing Ratio': NAN_value})
Missing_data.head()

def finding_zeros(frame):
    columns = frame.columns[:8]
    for i in columns:
        zeros = len(frame.loc[frame[i] == 0])
        print(f'The numbers of 0 values in {i} = {zeros}')
cond1 = df[(df['Insulin'] == 0) & (df['SkinThickness'] == 0) & (df['Pregnancies'] == 0) & (df['BloodPressure'] == 0) & (df['BMI'] == 0)].index
cond2 = df[(df['Insulin'] == 0) & (df['SkinThickness'] == 0) & (df['Pregnancies'] == 0) & (df['BloodPressure'] == 0)].index
Zeros_values = cond2.append(cond1)
df.drop(Zeros_values, inplace=True)
finding_zeros(df)
pass
mask = np.triu(df.corr())
pass
(bottom, top) = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
print('variable relationship')
pass
y = 'Insulin'
pass
pass
pass
print('This shows that the Higher Glucose. Higher Insuline')
lower = np.arange(26, 236, 5)
upper = np.arange(30, 205, 5)

def find_insuline(name, down, top):
    result = df.loc[(df['Insulin'] > 0) & (df['Glucose'] >= down) & (df['Glucose'] <= top)]['Insulin'].mean()
    result = np.round(result)
    Value.append(result)
Value = []
for (i, j) in zip(lower, upper):
    find_insuline('Insulin', i, j)
Value[1:9]

def find_insuline_index(name, down, top):
    Index = df[(df['Insulin'] == 0) & (df['Glucose'] >= down) & (df['Glucose'] <= top)].index.values
    Indexes.append(Index)
Indexes = []
for (i, j) in zip(lower, upper):
    find_insuline_index('Index', i, j)
Indexes[10:13]
for (i, j) in zip(np.arange(0, 36, 1), Value):
    df.loc[Indexes[i], 'Insulin'] = j
for i in np.arange(0, 4, 1):
    df.loc[Indexes[i], 'Insulin'] = 0
df.loc[Indexes[7], 'Insulin'] = 42
finding_zeros(df)
pass
y = 'SkinThickness'
pass
pass
pass
print('This shows that the greater BMI ,the greater SkinThickness (it maskes sense)')
i_skin = np.arange(15, 70, 5)
j_skin = np.arange(20, 75, 5)

def finding_skin(name, down, top):
    result = df.loc[(df['SkinThickness'] > 0) & (df['BMI'] >= down) & (df['BMI'] <= top)]['SkinThickness'].mean()
    result = np.round(result, 2)
    Skin_values.append(result)
Skin_values = []
for (i, j) in zip(i_skin, j_skin):
    finding_skin('Thickness', i, j)
Skin_values

def finding_skin_index(name, down, top):
    Index = df.loc[(df['SkinThickness'] == 0) & (df['BMI'] >= float(down)) & (df['BMI'] <= float(top))]['SkinThickness'].index.values
    Skin_Index.append(Index)
Skin_Index = []
for (i, j) in zip(i_skin, j_skin):
    finding_skin_index('Thickness', i, j)
Skin_Index[0:2]
for (i, j) in zip(np.arange(0, 10, 1), j_skin):
    df.loc[Skin_Index[i], 'SkinThickness'] = j
finding_zeros(df)
SKIN_BMI_ZERO = df[(df['SkinThickness'] == 0) & (df['BMI'] == 0)]
SKIN_BMI_ZERO
BLOOD_INSULINE_ZERO = df[(df['Glucose'] == 0) & (df['Insulin'] == 0)]
BLOOD_INSULINE_ZERO
BLOOD_INSULINE_ZERO = df[(df['Glucose'] == 0) & (df['Insulin'] == 0)].index.values
SKIN_BMI_ZERO = df[(df['SkinThickness'] == 0) & (df['BMI'] == 0)].index.values
df.drop(BLOOD_INSULINE_ZERO, inplace=True)
df.drop(SKIN_BMI_ZERO, inplace=True)
pass
pass
pass
df.drop(df[df['BloodPressure'] == 0].index.values, inplace=True)
NAN_value = df.isnull().sum() / len(df) * 100
Missing = NAN_value[NAN_value == 0].index.sort_values(ascending=False)
Missing_data = pd.DataFrame({'Missing Ratio': NAN_value})
Missing_data.head()

def boxplot(frame1, frame2, frame3):
    pass
    pass
    pass
    pass
boxplot(df.Pregnancies, df.Glucose, df.BloodPressure)
boxplot(df.SkinThickness, df.Insulin, df.BMI)
boxplot(df.DiabetesPedigreeFunction, df.Age, df.Outcome)
preg = df.loc[df['Pregnancies'] >= 15]['Pregnancies'].count()
glu = df.loc[df['Glucose'] < 40]['Glucose'].count()
blood_1 = df[df['BloodPressure'] < 40]['BloodPressure'].count()
blood_2 = df[df['BloodPressure'] > 100]['BloodPressure'].count()
blood = blood_1 + blood_2
skin = df[df['SkinThickness'] > 55]['SkinThickness'].count()
insu = df[df['Insulin'] > 380]['Insulin'].count()
bmi = df[df['BMI'] > 50]['BMI'].count()
dia = df[df['DiabetesPedigreeFunction'] > 1.2]['DiabetesPedigreeFunction'].count()
age = df[df['Age'] > 63]['Age'].count()
outliers = [preg, glu, blood, skin, insu, bmi, dia, age]
Outliers = pd.DataFrame(data=outliers, index=df.columns[0:8], columns=['Outliers'])
Outliers
preg_i = df.loc[df['Pregnancies'] >= 15]['Pregnancies'].index.values
glu_i = df.loc[df['Glucose'] < 40]['Glucose'].index.values
blood_1_i = df[df['BloodPressure'] < 40]['BloodPressure'].index.values
blood_2_i = df[df['BloodPressure'] > 100]['BloodPressure'].index.values
skin = df[df['SkinThickness'] > 55]['SkinThickness'].index.values
insu = df[df['Insulin'] > 380]['Insulin'].index.values
bmi = df[df['BMI'] > 50]['BMI'].index.values
dia = df[df['DiabetesPedigreeFunction'] > 1.2]['DiabetesPedigreeFunction'].index.values
age = df[df['Age'] > 63]['Age'].index.values
ind_out = [preg_i, glu_i, blood_1_i, blood_2_i, skin, insu, bmi, dia, age]
for i in ind_out:
    df_out = df.drop(i)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
X = df_out.drop('Outcome', axis=1)
y = df_out['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100)