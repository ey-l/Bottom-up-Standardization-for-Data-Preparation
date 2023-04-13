import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pass
import seaborn as sns
pass
import warnings
warnings.filterwarnings('ignore')
diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_data.head(10)
diabetes_data.sample(10)
df_name = diabetes_data.columns
diabetes_data.describe()
diabetes_data.shape
diabetes_data.info()
diabetes_data.describe().T
diabetes_data.columns
colsToModify = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in colsToModify:
    print(col + ' - ')
    print(diabetes_data[diabetes_data[col] == 0][col].value_counts())
print('Count of zero entries')
colsToModify = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
val = []
for col in colsToModify:
    val.append(len(diabetes_data[diabetes_data[col] == 0]))
zeroCount = pd.DataFrame(val, index=colsToModify, columns=['zeroCount'])
zeroCount
p = diabetes_data.hist(figsize=(20, 20))
for col in colsToModify:
    diabetes_data[col] = diabetes_data[col].replace(0, np.NaN)
    mean = int(diabetes_data[col].mean(skipna=True))
    diabetes_data[col] = diabetes_data[col].replace(np.NaN, mean)
diabetes_data.head()
diabetes_data.describe()
diabetes_data.duplicated().sum()
diabetes_data.isnull().sum()
p = diabetes_data.hist(figsize=(20, 20))
import matplotlib.style as style
style.available
style.use('seaborn-pastel')
labels = ['Healthy', 'Diabetic']
diabetes_data['Outcome'].value_counts().plot(kind='pie', labels=labels, subplots=True, autopct='%1.0f%%', labeldistance=1.2, figsize=(9, 9))
corr = diabetes_data.corr()
pass
pass
p = diabetes_data.hist(figsize=(20, 20))
from matplotlib.pyplot import figure, show
figure(figsize=(8, 6))
pass
ax.set_xticklabels(['Healthy', 'Diabetic'])
(healthy, diabetics) = diabetes_data['Outcome'].value_counts().values
print('Samples of diabetic people: ', diabetics)
print('Samples of healthy people: ', healthy)
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
from sklearn.model_selection import train_test_split
X = diabetes_data.iloc[:, 0:8]
y = diabetes_data.iloc[:, 8]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=21, p=2, metric='euclidean')