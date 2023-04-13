import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pima = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima.head()
pima.describe()
ind_var = pima[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
dep_var = pima.Outcome
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()