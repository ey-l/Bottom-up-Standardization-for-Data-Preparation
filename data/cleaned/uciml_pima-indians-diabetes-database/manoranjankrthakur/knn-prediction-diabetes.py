import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
import matplotlib.pyplot as plt
import seaborn as sns
df.head()
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
sns.set_style('whitegrid')
sns.pairplot(df, hue='Outcome', size=9)

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)
df.isnull().sum()
x = df.iloc[:, 0:8]
y = df.iloc[:, 8]
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=0, test_size=0.2)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
Classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')