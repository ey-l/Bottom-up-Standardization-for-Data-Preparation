import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
X = df.iloc[:, 0:8]
y = df.iloc[:, -1]
X
df.describe()
import matplotlib.pyplot as plt
import seaborn as sns
matrix = np.triu(df.corr())
sns.heatmap(df.corr(), annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap='YlOrRd', mask=matrix)
feature = X.columns
dfzero = (X[feature] == 0).sum()
dfzero
X[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = X[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
X.isnull().sum()
X['Glucose'].fillna(X['Glucose'].mean(), inplace=True)
X['BMI'].fillna(X['BMI'].mean(), inplace=True)
X['Pregnancies'].fillna(X['Pregnancies'].mean(), inplace=True)
X['BloodPressure'].fillna(X['BloodPressure'].mean(), inplace=True)
X['SkinThickness'].fillna(X['SkinThickness'].mean(), inplace=True)
X['Insulin'].fillna(X['Insulin'].mean(), inplace=True)
X.describe()
X.isnull().sum()
X1 = X['Glucose']
Y1 = X['Age']
plt.scatter(X1, Y1)
plt.xlabel('Glucose Level')
plt.ylabel('Age')
plt.title(label='Age Vs Glucose Chart')

X['HighRisk'] = np.where(X['SkinThickness'] > 23, 1, 0)
X['AgeBracket'] = np.select([X.Age >= 50, (X.Age >= 30) & (X.Age < 50), X.Age < 30], [3, 2, 1])
X.head()
X = X.drop(columns=['Age', 'SkinThickness'])
X.head()
from sklearn.model_selection import train_test_split
(Xtrain, Xtest, ytrain, ytest) = train_test_split(X, y, test_size=0.2, random_state=0)
Xtrain.shape
Xtest.shape
Xtrain.head()
from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
Xtrain.iloc[:, :6] = Sc.fit_transform(Xtrain.iloc[:, :6])
Xtest.iloc[:, :6] = Sc.transform(Xtest.iloc[:, :6])
Xtrain
from sklearn.svm import SVC
SVC_Classifier = SVC()