import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.sample(5)
dataset.shape
dataset.describe()
dataset.info()
dataset.loc[dataset['Pregnancies'] == max(dataset['Pregnancies'])]
diabetic_patients = len(dataset[dataset['Outcome'] == 1])
non_diabetic_patients = dataset.shape[0] - diabetic_patients
print(diabetic_patients, non_diabetic_patients)
print(len(dataset[dataset['Glucose'] == 0]))
dataset[dataset['Glucose'] == 0]
import matplotlib.pyplot as plt
from scipy.stats import norm
glucose_mean = np.mean(dataset['Glucose'])
glucose_std = np.std(dataset['Glucose'])
glucose_median = np.median(dataset['Glucose'].sort_values())
plt.plot(dataset['Glucose'].sort_values(), norm.pdf(dataset['Glucose'].sort_values(), glucose_mean, glucose_std))

print(glucose_mean, glucose_median)
dataset['Glucose'] = dataset['Glucose'].replace(0, glucose_mean)
dataset[dataset['BloodPressure'] == 0]
bloodPressure_mean = np.mean(dataset['BloodPressure'])
bloodPressure_std = np.std(dataset['BloodPressure'])
bloodPressure_median = np.median(dataset['BloodPressure'].sort_values())
plt.plot(dataset['BloodPressure'].sort_values(), norm.pdf(dataset['BloodPressure'].sort_values(), bloodPressure_mean, bloodPressure_std))

print(bloodPressure_mean, bloodPressure_median)
dataset['BloodPressure'] = dataset['BloodPressure'].replace(0, bloodPressure_mean)
st_size = len(dataset[dataset['SkinThickness'] == 0])
print(st_size, dataset.shape[0])
st_mean = np.mean(dataset['SkinThickness'])
st_std = np.std(dataset['SkinThickness'])
st_median = np.median(dataset['SkinThickness'].sort_values())
dataset['SkinThickness'] = dataset['SkinThickness'].replace(0, st_mean)
in_mean = np.mean(dataset['Insulin'])
in_std = np.std(dataset['Insulin'])
in_median = np.median(dataset['Insulin'].sort_values())
print(in_mean, in_median)
pdf = norm.pdf(dataset['Insulin'].sort_values(), in_mean, in_std)
plt.plot(dataset['Insulin'].sort_values(), pdf)

print(dataset.shape[0], len(dataset[dataset['Insulin'] == 0]))
dataset['Insulin'] = dataset['Insulin'].replace(0, in_median)
BMI_mean = np.mean(dataset['BMI'])
BMI_std = np.std(dataset['BMI'])
BMI_median = np.median(dataset['BMI'].sort_values())
dataset['BMI'] = dataset['BMI'].replace(0, BMI_mean)
dataset.describe()
labels = ['Non-Diabatic', 'Diabatic']
explode = (0, 0.1)
sizes = [non_diabetic_patients, diabetic_patients]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode, startangle=90)

print(sizes)
dataset.hist(bins=10, figsize=(10, 10))

import seaborn as sns
corrmat = dataset.corr()
corr_features = corrmat.index
plt.figure(figsize=(9, 9))
sns.heatmap(corrmat, annot=True, cmap='RdYlGn')
target = 'Outcome'
y = dataset[target]
X = dataset.drop(target, axis=1)
print(X.shape, y.shape)
'\nShould you apply feature scaling or any preprocessing algorithm before or after\nsplitting the data?\nWhen you fit the standard scaler on the whole dataset, \ninformation from the test set is used to normalize the training set. \nThis is a common case of "data leakage", \nwhich means that information from the test set is used while training the model. \nThis often results in overestimates of the model\'s performance. \nTherefore, apply it after splitting\n'
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler