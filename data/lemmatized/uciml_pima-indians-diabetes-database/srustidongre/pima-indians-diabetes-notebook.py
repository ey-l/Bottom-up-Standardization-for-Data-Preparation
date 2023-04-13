import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.info()
data.describe().transpose()
data.duplicated().sum()
data.isnull().sum()
data.nunique()
quantiles = data.quantile(np.arange(0.1, 1, 0.1)).transpose()
quantiles
top_percentiles = data.quantile(np.arange(0.9, 1, 0.01)).transpose()
top_percentiles
bottom_percentiles = data.quantile(np.arange(0, 0.1, 0.01)).transpose()
bottom_percentiles
data.min()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_histograms(df, variables, n_rows, n_cols):
    pass
    for (i, var_name) in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        df[var_name].hist(bins=10, ax=ax, color='purple')
        ax.set_title(var_name + ' Distribution')
    fig.tight_layout()
draw_histograms(data, data.columns, 8, 3)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_box_plots(df, variables, n_rows, n_cols):
    pass
    pass
    for (i, var_name) in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        pass
        ax.set_title(var_name + ' Box Polt')
    fig.tight_layout()
    pass
draw_box_plots(data, data.columns, 11, 3)
data[data['Glucose'] == 0]
data[data['BloodPressure'] == 0]
data[data['Insulin'] == 0].head()
data[data['Insulin'] == 0].head()
data[data['SkinThickness'] == 0].head()
data[data['BMI'] == 0].head()
data.columns
data['Glucose'] = np.where(data['Glucose'] == 0, data['Glucose'].quantile(0.02), data['Glucose'])
data['BloodPressure'] = np.where(data['BloodPressure'] == 0, data['BloodPressure'].quantile(0.05), data['BloodPressure'])
data['BMI'] = np.where(data['BMI'] == 0, data['BMI'].quantile(0.02), data['BMI'])
data.quantile(np.arange(0.01, 0.1, 0.01)).transpose()
pass
pass
pass
pass
no_diab = data[data['Outcome'] == 0]
diab = data[data['Outcome'] == 1]
pass
pass
ax = no_diab['BMI'].plot.hist(bins=20, alpha=0.5, color='green')
ax = diab['BMI'].plot.hist(bins=12, alpha=0.5, color='purple')
pass
pass
ax = no_diab['BloodPressure'].plot.hist(bins=20, alpha=0.5, color='green')
ax = diab['BloodPressure'].plot.hist(bins=12, alpha=0.5, color='purple')
pass
pass
ax = no_diab['SkinThickness'].plot.hist(bins=20, alpha=0.5, color='green')
ax = diab['SkinThickness'].plot.hist(bins=12, alpha=0.5, color='purple')
pass
pass
ax = no_diab['Insulin'].plot.hist(bins=20, alpha=0.5, color='green')
ax = diab['Insulin'].plot.hist(bins=12, alpha=0.5, color='purple')
pass
pass
ax = no_diab['Age'].plot.hist(bins=20, alpha=0.5, color='green')
ax = diab['Age'].plot.hist(bins=12, alpha=0.5, color='purple')
pass
pass
ax = no_diab['Pregnancies'].plot.hist(bins=20, alpha=0.5, color='green')
ax = diab['Pregnancies'].plot.hist(bins=12, alpha=0.5, color='purple')
pass
pass
data.columns
data.head()
data.columns
X = data[['Pregnancies', 'Glucose', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction']]
y = data['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42, test_size=0.2)
print('\nX_train.shape : ', X_train.shape, '\nX_test.shape  : ', X_test.shape, '\ny_train.shape : ', y_train.shape, '\ny_test.shape  : ', y_test.shape)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data.sort_values(by='VIF', ascending=False))
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=42)