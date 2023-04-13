import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
print('Setup Complete')
filepath = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
dd = pd.read_csv(filepath)
dd.describe()
req_data = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'Pregnancies']
ddf = dd[req_data]
ddf.info()
ddf.head()
ddf.isnull().sum()
y = ddf.Outcome
features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Pregnancies']
X = ddf[features]
(train_X, val_X, train_y, val_y) = train_test_split(X, y, random_state=7)
df = RandomForestClassifier(random_state=7)