import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.tail()
df.dtypes
df.info()
print(df.isnull().sum())
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange']
for (i, col) in enumerate(df.columns):
    pass
    pass
    pass
    pass
    pass
    pass
pass
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange']
for (i, col) in enumerate(df.columns):
    pass
    pass
    pass
    pass
    pass
outcome_counts = df['Outcome'].value_counts()
pass
pass
pass
pass
pass
pass
pass
pass
pass
import warnings
warnings.filterwarnings('ignore')
pass
pass
for col in df.columns[:-1]:
    pass
    pass
    pass
    pass
    pass
pass
corr_matrix = df.corr()
pass
pass
pass
fig = px.parallel_coordinates(df, color='Outcome', dimensions=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
pass
pass
pass
custom_palette = ['blue', 'red']
pass
pass