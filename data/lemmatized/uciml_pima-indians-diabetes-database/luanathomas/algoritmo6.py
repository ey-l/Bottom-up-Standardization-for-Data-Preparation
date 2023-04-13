import pandas as pd
import numpy as np
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
origem = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
dados = pd.read_csv(origem, header=None)
dados = dados.drop(0)
col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
dados.columns = col_names
dados['Pregnancies'] = pd.to_numeric(dados['Pregnancies'], errors='coerce')
dados['Glucose'] = pd.to_numeric(dados['Glucose'], errors='coerce')
dados['BloodPressure'] = pd.to_numeric(dados['BloodPressure'], errors='coerce')
dados['SkinThickness'] = pd.to_numeric(dados['SkinThickness'], errors='coerce')
dados['Insulin'] = pd.to_numeric(dados['Insulin'], errors='coerce')
dados['BMI'] = pd.to_numeric(dados['BMI'], errors='coerce')
dados['DiabetesPedigreeFunction'] = pd.to_numeric(dados['DiabetesPedigreeFunction'], errors='coerce')
dados['Age'] = pd.to_numeric(dados['Age'], errors='coerce')
dados['Outcome'] = pd.to_numeric(dados['Outcome'], errors='coerce')
x = dados[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = dados['Outcome']
SEED = 50
np.random.seed(SEED)
(treino_x_raw, teste_x_raw, treino_y, teste_y) = train_test_split(x, y, test_size=0.2, stratify=y)
print('Treinaremos com %d elementos e testaremos com %d elementos' % (len(treino_x_raw), len(teste_x_raw)))
scaler = Normalizer()
treino_x = scaler.fit_transform(treino_x_raw)
teste_x = scaler.transform(teste_x_raw)
col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
features = dados[col_names]
ct = ColumnTransformer([('somename', Normalizer(), ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])], remainder='passthrough')
dados = ct.fit_transform(features)
labels = []
for label in y:
    if label in labels:
        continue
    else:
        labels.append(label)
epsilon = 2
classifier = KNeighborsClassifier(n_neighbors=8)