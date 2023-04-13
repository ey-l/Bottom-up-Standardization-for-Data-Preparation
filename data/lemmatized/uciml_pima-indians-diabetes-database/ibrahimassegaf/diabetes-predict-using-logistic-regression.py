import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import sklearn.metrics as mt
dataframe = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataframe.head()
dataframe.info()
dataframe.describe()
x = np.array(['0', '1'])
y = dataframe['Outcome'].value_counts()
pass
print(y)
atribut = dataframe[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
label = dataframe.Outcome
(xTrain, xTest, yTrain, yTest) = train_test_split(atribut, label, test_size=0.2, random_state=0)
xTrain.describe()
xTest.describe()
model = LogisticRegression(solver='lbfgs')