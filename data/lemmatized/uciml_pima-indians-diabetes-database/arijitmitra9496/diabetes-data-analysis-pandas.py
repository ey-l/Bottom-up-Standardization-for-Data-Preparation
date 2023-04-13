import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_data.head()
print(len(diabetes_data.columns))
print(len(diabetes_data))
diabetes_data.isnull().sum()
diabetes_data.info()
diabetes_data.describe()
diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
diabetes_data.info()
diabetes_data.isnull().sum()
diabetes_data[diabetes_data.isnull().any(axis=1)]
diabetes_data.dtypes