import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
db = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
dados = pd.read_csv(db, header=None)
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
raio = [0.03, 0.06, 0.09]
labels = []
for label in y:
    if label in labels:
        continue
    else:
        labels.append(label)

def query(db, r, pontoCentral, label):
    total = 0
    feature1a = pontoCentral[0]
    feature2a = pontoCentral[1]
    feature3a = pontoCentral[2]
    feature4a = pontoCentral[3]
    feature5a = pontoCentral[4]
    feature6a = pontoCentral[5]
    feature7a = pontoCentral[6]
    feature8a = pontoCentral[7]
    for i in range(len(db)):
        feature1b = db[i][0]
        feature2b = db[i][1]
        feature3b = db[i][2]
        feature4b = db[i][3]
        feature5b = db[i][4]
        feature6b = db[i][5]
        feature7b = db[i][6]
        feature8b = db[i][7]
        classe = db[i][8]
        distancia = math.sqrt((feature1b - feature1a) ** 2 + (feature2b - feature2a) ** 2 + (feature3b - feature3a) ** 2 + (feature4b - feature4a) ** 2 + (feature5b - feature5a) ** 2 + (feature6b - feature6a) ** 2 + (feature7b - feature7a) ** 2 + (feature8b - feature8a) ** 2)
        if distancia <= r and classe == label:
            total += 1
    return total

def algoritmo2(db, labels, pontoCentral, raio):
    probabilidades = []
    for label in labels:
        quant_vizinhos = query(db, raio, pontoCentral, label)
        probabilidades.append(math.e ** (epsilon * quant_vizinhos / 2))
    return probabilidades
epsilon = 2
total_elementos = len(teste_x)
for k in range(len(raio)):
    acertos = 0
    r = raio[k]
    print()
    print('Raio %.2f' % r)
    for i in range(len(teste_x)):
        pontoCentral = [teste_x[i][0], teste_x[i][1], teste_x[i][2], teste_x[i][3], teste_x[i][4], teste_x[i][5], teste_x[i][6], teste_x[i][7]]
        algoritmo2_ = algoritmo2(dados, labels, pontoCentral, r)
        probabilidades = []
        total = 0
        for j in range(len(algoritmo2_)):
            total += algoritmo2_[j]
        for j in range(len(algoritmo2_)):
            porcentagem = algoritmo2_[j] * 100 / total
            probabilidades.append(porcentagem)
        for l in range(len(probabilidades)):
            print('Porcentagem do label %d: %.2f%%' % (labels[l], probabilidades[l]))
        print()
    print()
    print()
    print('______________________________________________________________________________________________________________________________________')