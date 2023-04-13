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
raio = 0.02
labels = []
for label in y:
    if label in labels:
        continue
    else:
        labels.append(label)
lista_de_nodos = []
for i in range(len(treino_x)):
    lista_de_nodos.append(i)

def arestasVizinhas(db, raio, pontoCentral, lista_de_arcos, i):
    feature1a = pontoCentral[0]
    feature2a = pontoCentral[1]
    feature3a = pontoCentral[2]
    feature4a = pontoCentral[3]
    feature5a = pontoCentral[4]
    feature6a = pontoCentral[5]
    feature7a = pontoCentral[6]
    feature8a = pontoCentral[7]
    for j in range(len(db)):
        feature1b = db[j][0]
        feature2b = db[j][1]
        feature3b = db[j][2]
        feature4b = db[j][3]
        feature5b = db[j][4]
        feature6b = db[j][5]
        feature7b = db[j][6]
        feature8b = db[j][7]
        distancia = math.sqrt((feature1b - feature1a) ** 2 + (feature2b - feature2a) ** 2 + (feature3b - feature3a) ** 2 + (feature4b - feature4a) ** 2 + (feature5b - feature5a) ** 2 + (feature6b - feature6a) ** 2 + (feature7b - feature7a) ** 2 + (feature8b - feature8a) ** 2)
        if distancia <= raio * 2 and i != j:
            lista_de_arcos.append([i, j])
    return lista_de_arcos
lista_de_arcos = []
for i in range(len(treino_x)):
    pontoCentral = [treino_x[i][0], treino_x[i][1], treino_x[i][2], treino_x[i][3], treino_x[i][4], treino_x[i][5], treino_x[i][6], treino_x[i][7]]
    arestasVizinhas(treino_x, raio, pontoCentral, lista_de_arcos, i)

def cria_grafo(lista_de_nodos, lista_de_arcos):
    grafo = {}
    for nodo in lista_de_nodos:
        grafo[nodo] = []
    for arco in lista_de_arcos:
        grafo[arco[0]].append(arco[1])
    return grafo
grafo = cria_grafo(lista_de_nodos, lista_de_arcos)

def verifica_distancia(treino_x, raio, pontoCentral, elementoAnalisado):
    feature1a = pontoCentral[0]
    feature2a = pontoCentral[1]
    feature3a = pontoCentral[2]
    feature4a = pontoCentral[3]
    feature5a = pontoCentral[4]
    feature6a = pontoCentral[5]
    feature7a = pontoCentral[6]
    feature8a = pontoCentral[7]
    feature1b = elementoAnalisado[0]
    feature2b = elementoAnalisado[1]
    feature3b = elementoAnalisado[2]
    feature4b = elementoAnalisado[3]
    feature5b = elementoAnalisado[4]
    feature6b = elementoAnalisado[5]
    feature7b = elementoAnalisado[6]
    feature8b = elementoAnalisado[7]
    distancia = math.sqrt((feature1b - feature1a) ** 2 + (feature2b - feature2a) ** 2 + (feature3b - feature3a) ** 2 + (feature4b - feature4a) ** 2 + (feature5b - feature5a) ** 2 + (feature6b - feature6a) ** 2 + (feature7b - feature7a) ** 2 + (feature8b - feature8a) ** 2)
    if distancia <= raio * 2:
        return True
    else:
        return False

def add_subgrafo(treino_x, raio, pontoCentral, nodos_visitados, lista_de_nodos, subgrafo):
    for elemento in range(len(treino_x)):
        if verifica_distancia(treino_x, raio, pontoCentral, treino_x[elemento]) and elemento not in nodos_visitados and (elemento not in subgrafo):
            subgrafo.append(elemento)
    return subgrafo

def verifica_todos_visitados(nodos_visitados, subgrafo):
    todos_visitados = True
    for elemento in subgrafo:
        if elemento not in nodos_visitados:
            todos_visitados = False
    return todos_visitados
nodos_visitados = []
subgrafos = []
for nodo in range(len(lista_de_nodos)):
    if lista_de_nodos[nodo] not in nodos_visitados:
        nodos_visitados.append(lista_de_nodos[nodo])
        subgrafo = []
        subgrafo.append(lista_de_nodos[nodo])
        add_subgrafo(treino_x, raio, treino_x[nodo], nodos_visitados, lista_de_nodos, subgrafo)
        while verifica_todos_visitados(nodos_visitados, subgrafo) == False:
            for elemento in subgrafo:
                if elemento not in nodos_visitados:
                    nodos_visitados.append(elemento)
                    add_subgrafo(treino_x, raio, treino_x[elemento], nodos_visitados, lista_de_nodos, subgrafo)
        subgrafos.append(subgrafo)

def query(db, r, pontoCentral, label, classes, todos_vizinhos):
    total = 0
    vizinhos_elemento = []
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
            vizinhos_elemento.append(db[i])
    if label in classes:
        esta_na_lista = False
        for vizinho in vizinhos_elemento:
            for v in todos_vizinhos[classes.index(label)]:
                if vizinho.all() == v.all():
                    esta_na_lista = True
            if esta_na_lista == False:
                todos_vizinhos[classes.index(label)].append(vizinho)
                total += 1
    else:
        todos_vizinhos.append(vizinhos_elemento)
        classes.append(label)
        total += len(vizinhos_elemento)
    return total

def algoritmo1(db, labels, pontoCentral, raio, ruido_0, ruido_1):
    classes = []
    todos_vizinhos = []
    quant = [0, 0]
    for label in labels:
        if label == 0:
            elementos = []
            a = query(db, raio, pontoCentral, label, classes, todos_vizinhos)
            a_ruido = a + ruido_0
            quant[classes.index(label)] += a_ruido
        else:
            elementos = []
            a = query(db, raio, pontoCentral, label, classes, todos_vizinhos)
            a_ruido = a + ruido_1
            quant[classes.index(label)] += a_ruido
    label = classes[np.argmax(quant)]
    return label
epsilon = 2 / len(subgrafos)
ruido_0 = np.random.laplace(0, 1 / epsilon)
ruido_1 = np.random.laplace(0, 1 / epsilon)
labels_algoritmo3 = []
total_label_0 = 0
total_label_1 = 0
total_labels = 0
for grafo in subgrafos:
    labels_grafo = []
    for instancia_teste in grafo:
        label_instancia = algoritmo1(dados, labels, dados[instancia_teste], raio, ruido_0, ruido_1)
        if label_instancia == 0:
            total_label_0 += 1
        else:
            total_label_1 += 1
        total_labels += 1
        labels_grafo.append(label_instancia)
    labels_algoritmo3.append(labels_grafo)
print('Total de labels 0: %.2f%%' % (100 * total_label_0 / total_labels))
print('Total de labels 1: %.2f%%' % (100 * total_label_1 / total_labels))
labels_algoritmo3

def algoritmo1(db, labels, pontoCentral, raio, ruido):
    classes = []
    todos_vizinhos = []
    quant = [0, 0]
    for label in labels:
        elementos = []
        a = query(db, raio, pontoCentral, label, classes, todos_vizinhos)
        a_ruido = a + ruido
        quant[classes.index(label)] += a_ruido
    label = classes[np.argmax(quant)]
    return label
epsilon = 2 / len(subgrafos)
ruido = np.random.laplace(0, 1 / epsilon)
labels_algoritmo3 = []
total_label_0 = 0
total_label_1 = 0
total_labels = 0
for grafo in subgrafos:
    labels_grafo = []
    for instancia_teste in grafo:
        label_instancia = algoritmo1(dados, labels, dados[instancia_teste], raio, ruido)
        if label_instancia == 0:
            total_label_0 += 1
        else:
            total_label_1 += 1
        total_labels += 1
        labels_grafo.append(label_instancia)
    labels_algoritmo3.append(labels_grafo)
print('Total de labels 0: %.2f%%' % (100 * total_label_0 / total_labels))
print('Total de labels 1: %.2f%%' % (100 * total_label_1 / total_labels))
labels_algoritmo3

def algoritmo1(db, labels, pontoCentral, raio, ruido):
    classes = []
    todos_vizinhos = []
    quant = [0, 0]
    for label in labels:
        elementos = []
        a = query(db, raio, pontoCentral, label, classes, todos_vizinhos)
        a_ruido = a + np.random.laplace(0, 1 / epsilon)
        quant[classes.index(label)] += a_ruido
    label = classes[np.argmax(quant)]
    return label
epsilon = 2 / len(subgrafos)
labels_algoritmo3 = []
total_label_0 = 0
total_label_1 = 0
total_labels = 0
for grafo in subgrafos:
    labels_grafo = []
    for instancia_teste in grafo:
        label_instancia = algoritmo1(dados, labels, dados[instancia_teste], raio, ruido)
        if label_instancia == 0:
            total_label_0 += 1
        else:
            total_label_1 += 1
        total_labels += 1
        labels_grafo.append(label_instancia)
    labels_algoritmo3.append(labels_grafo)
print('Total de labels 0: %.2f%%' % (100 * total_label_0 / total_labels))
print('Total de labels 1: %.2f%%' % (100 * total_label_1 / total_labels))
labels_algoritmo3

def algoritmo1(db, labels, pontoCentral, raio, ruido):
    classes = []
    todos_vizinhos = []
    quant = [0, 0]
    for label in labels:
        elementos = []
        a = query(db, raio, pontoCentral, label, classes, todos_vizinhos)
        quant[classes.index(label)] += a
    label = classes[np.argmax(quant)]
    return label
labels_algoritmo3 = []
total_label_0 = 0
total_label_1 = 0
total_labels = 0
for grafo in subgrafos:
    labels_grafo = []
    for instancia_teste in grafo:
        label_instancia = algoritmo1(dados, labels, dados[instancia_teste], raio, ruido)
        if label_instancia == 0:
            total_label_0 += 1
        else:
            total_label_1 += 1
        total_labels += 1
        labels_grafo.append(label_instancia)
    labels_algoritmo3.append(labels_grafo)
print('Total de labels 0: %.2f%%' % (100 * total_label_0 / total_labels))
print('Total de labels 1: %.2f%%' % (100 * total_label_1 / total_labels))
labels_algoritmo3