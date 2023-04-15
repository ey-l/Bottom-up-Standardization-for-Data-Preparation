import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

diabetes_base_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print('\n'.join(diabetes_base_data.columns.values))
diabetes_base_data.head()
diabetes_base_data.shape
diabetes_base_data.dtypes
diabetes_base_data.describe()
diabetes_data = diabetes_base_data.copy()
diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
diabetes_data.describe()
mean_value_BloodPressure = diabetes_data['BloodPressure'].mean()
print(mean_value_BloodPressure)
diabetes_data.loc[diabetes_data['BloodPressure'].isnull(), 'BloodPressure'] = mean_value_BloodPressure
diabetic = diabetes_data.loc[diabetes_data['Outcome'] == 1, 'Outcome'].count()
print(diabetic)
healthy = diabetes_data.loc[diabetes_data['Outcome'] == 0, 'Outcome'].count()
print(healthy)
objects = ('diabetic', 'healthy')
y_pos = np.arange(len(objects))
count = [diabetic, healthy]
plt.barh(y_pos, count, align='center', alpha=0.5, color=['red', 'green'])
plt.yticks(y_pos, objects)
plt.title('Count of Outcome')
for (index, value) in enumerate(count):
    plt.text(value, index, str(value))

for (i, count) in enumerate(diabetes_data[['Outcome', 'Pregnancies', 'Age']]):
    plt.figure(i)
    sns.countplot(data=diabetes_data, x=count, hue='Outcome')
diabetes_data.loc[diabetes_data['Outcome'] == 1].describe()
diabetes_data.loc[diabetes_data['Outcome'] == 0].describe()
plt.figure(figsize=(15, 5))
diabetes_data.corr()['Outcome'].drop('Outcome').sort_values(ascending=False).plot(kind='bar')
x1 = diabetes_data.loc[diabetes_data.Outcome == 1, 'BMI']
x2 = diabetes_data.loc[diabetes_data.Outcome == 0, 'BMI']
kwargs = dict(alpha=0.3, bins=25)
plt.hist(x1, **kwargs, color='r', label='diabetic')
plt.hist(x2, **kwargs, color='g', label='healthy')
plt.gca().set(title='BMI', xlabel='BMI values', ylabel='count')
plt.xlim(20, 60)
plt.legend()
x1 = diabetes_data.loc[diabetes_data.Outcome == 1, 'Age']
x2 = diabetes_data.loc[diabetes_data.Outcome == 0, 'Age']
kwargs = dict(alpha=0.3, bins=20)
plt.hist(x1, **kwargs, color='r', label='diabetic')
plt.hist(x2, **kwargs, color='g', label='healthy')
plt.gca().set(title='Age', xlabel='Age', ylabel='count')
plt.xlim(25, 100)
plt.legend()
x1 = diabetes_data.loc[diabetes_data.Outcome == 1, 'SkinThickness']
x2 = diabetes_data.loc[diabetes_data.Outcome == 0, 'SkinThickness']
kwargs = dict(alpha=0.3, bins=20)
plt.hist(x1, **kwargs, color='r', label='diabetic')
plt.hist(x2, **kwargs, color='g', label='healthy')
plt.gca().set(title='Skin Thickness', xlabel='SkinThickness', ylabel='count')
plt.xlim(0, 100)
plt.legend()
x1 = diabetes_data.loc[diabetes_data.Outcome == 1, 'Pregnancies']
x2 = diabetes_data.loc[diabetes_data.Outcome == 0, 'Pregnancies']
kwargs = dict(alpha=0.3, bins=15)
plt.hist(x1, **kwargs, color='r', label='diabetic')
plt.hist(x2, **kwargs, color='g', label='healthy')
plt.gca().set(title='Pregnancies', xlabel='number of Pregnancies', ylabel='count')
plt.xlim(0, 18)
plt.legend()
diabetes_data.loc[diabetes_data['BMI'] >= 30, 'BMI'].count()
diabetic2 = diabetes_data.loc[diabetes_data['Outcome'] == 1]
obese = diabetic2.loc[diabetic2['BMI'] >= 30, 'BMI'].count()
print(obese)
df = pd.DataFrame(diabetes_data)
df.quantile([0.1, 0.25, 0.5, 0.75], axis=0)
df2 = pd.DataFrame(diabetic2)
df2.quantile([0.1, 0.25, 0.5, 0.75], axis=0)