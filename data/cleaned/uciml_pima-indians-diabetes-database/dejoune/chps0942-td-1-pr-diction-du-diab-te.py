import pandas as pd
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print('Shape (n_rows,n_columns) of dataframe:', df.shape)
df.head()
df2 = df[['Outcome', 'Pregnancies', 'Insulin']].head()
df2
df3 = df[['Age']].head()
df3
print(df[df.BMI > 30].shape)
print('The number of rows where BMI>30 = ', df[df.BMI > 30].shape[0])
df[df.BMI < 10].head()
df.BMI > 30
df.isnull().sum()
df.notnull().sum()
df.columns
df.dtypes
df.describe()
df.Outcome.value_counts()
df[df.Outcome == 1].SkinThickness.mean()
df[df['Outcome'] == 1].SkinThickness.mean()
df[df['Outcome'] == 1]['SkinThickness'].mean()
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(df[df.Outcome == 1].BloodPressure, df[df.Outcome == 1].Glucose, label='Diabete', color='r', marker='o', s=5)
plt.scatter(df[df.Outcome == 0].BloodPressure, df[df.Outcome == 0].Glucose, label='No Diabete', color='b', marker='x', s=5)
plt.legend()
plt.xlabel('BloodPressure')
plt.ylabel('Glucose')
dimension = max([len(c) for c in df.columns])
for c in df.columns:
    print("Pour l'attribut {quoi!r:<{dim}} il y a {combien:>4} valeurs 0.".format(quoi=c, combien=df[df[c] == 0][c].count(), dim=dimension + 2))
for c in df.columns:
    plt.figure()
    plt.hist(df[c], bins=15)
    plt.xlabel(c)
    plt.ylabel('Fréquence')

plt.figure
for c in df.columns:
    plt.scatter(df[df.Outcome == 1].BloodPressure, df[df.Outcome == 1].Glucose, label='Diabete', color='r', s=3)
    plt.xlabel('Pression sanguine')
    plt.ylabel('Sucre')
plt.figure()
plt.hist(df[df.Outcome == 1]['Age'], bins=15, label='Diabetes', color='r', alpha=0.2)
plt.hist(df[df.Outcome == 0]['Age'], bins=15, label='No Diabetes', color='b', alpha=0.2)
plt.xlabel('Age')
plt.ylabel('Frequence')
plt.legend()

import numpy as np
df['PredictedOutcome'] = np.where(df.Age < 100, 0, 1)
N_correct = df[df.PredictedOutcome == df.Outcome].shape[0]
N_total = df.shape[0]
accuracy = N_correct / N_total
print('number of correct examples =', N_correct)
print('number of examples in total =', N_total)
print('accuracy =', accuracy)
donnees = df
donnees.drop('Insulin', axis=1, inplace=True)
df.columns
import sklearn
from sklearn.model_selection import train_test_split
(train, test) = train_test_split(donnees, test_size=0.3, random_state=0)
train.describe()
import numpy as np

def imputeColumns(dataset):
    """ Pour chacune des colonnes du dataset,
        mise à jour des valeurs à zero par la moyenne de ses valeurs non nulles.
    """
    columnsToImpute = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
    for c in columnsToImpute:
        avgOfCol = dataset[dataset[c] > 0][[c]].mean()
        dataset[c + '_imputed'] = np.where(dataset[[c]] != 0, dataset[[c]], avgOfCol)
imputeColumns(train)
imputeColumns(test)
train[['Glucose', 'Glucose_imputed']].head()
X_train = train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y_train = train[['Outcome']]
X_test = test[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y_test = test[['Outcome']]
X_train.columns
Y_train.describe()
Y_test.describe()
from sklearn import tree
mon_arbre_de_decision = tree.DecisionTreeClassifier(random_state=0)