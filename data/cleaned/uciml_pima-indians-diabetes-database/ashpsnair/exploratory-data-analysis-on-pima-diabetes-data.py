import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data.head()
print('Data shape=' + str(data.shape))
data.info()
print('Null Values for this dataset')
print(data.isnull().sum())
data.describe()
remove = ['Glucose', 'BMI', 'SkinThickness', 'BloodPressure', 'Age', 'Insulin']
new_data = data
for i in remove:
    new_data = new_data[new_data[i] != 0]
print('No. of rows removed = ' + str(len(data) - len(new_data)))
modifs = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
data[modifs] = data[modifs].replace(0, np.NaN)
missing_values = (data.isnull().sum() / len(data) * 100).round(2)
print(missing_values)
data.hist(bins=25, figsize=(20, 15))
for column in data:
    plt.figure()
    data.boxplot([column])
sns.pairplot(data=data)
corr = data.corr(method='spearman')
corr
factors = ['Age', 'Pregnancies', 'BMI', 'SkinThickness', 'Glucose', 'Insulin', 'Outcome']
data[factors].corr()
sns.heatmap(data[factors].corr(), annot=True, cmap='Reds')

sns.pairplot(data=data, vars=factors, hue='Outcome')