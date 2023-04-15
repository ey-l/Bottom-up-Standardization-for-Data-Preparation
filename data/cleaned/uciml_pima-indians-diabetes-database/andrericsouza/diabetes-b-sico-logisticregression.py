import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
dados = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dados.head()
dados.shape
dados.columns
dados.dtypes
dados.describe(include='all').T.round(2)
dados.isna().sum()
dados.info()
dados.corr()
plt.figure(figsize=(15, 10))
mascara = np.triu(np.ones(dados.corr().shape)).astype(np.bool)
sns.heatmap(dados.corr().round(2), annot=True, cmap='BuPu', mask=mascara)

dados.nunique()

def f_grafico(df, coluna, target):
    dados = df
    target = target
    coluna = coluna
    analise = dados[coluna]
    (f, (ax_box, ax_hist)) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.2, 0.8)}, figsize=(15, 5))
    f.suptitle(pd.DataFrame(analise).columns[0])
    sns.boxplot(analise, ax=ax_box)
    sns.distplot(analise, ax=ax_hist, bins=3, color='g')
    ax_box.set(xlabel='')

    plt.figure(figsize=(15, 10))
    sns.distplot(analise)

    g = sns.FacetGrid(dados, col=target, height=5)
    g = g.map(sns.distplot, coluna, kde=False, color='c')
    print(pd.DataFrame(analise).describe().round(2).T)
f_grafico(dados, 'Pregnancies', 'Outcome')
f_grafico(dados, 'Glucose', 'Outcome')
f_grafico(dados, 'BloodPressure', 'Outcome')
f_grafico(dados, 'SkinThickness', 'Outcome')
f_grafico(dados, 'Insulin', 'Outcome')
f_grafico(dados, 'BMI', 'Outcome')
f_grafico(dados, 'DiabetesPedigreeFunction', 'Outcome')
f_grafico(dados, 'Age', 'Outcome')
dados['total'] = 1
dados.groupby('Outcome', as_index=False)['total'].count()
dados['total'] = 1
x = dados.groupby('Outcome', as_index=False)['total'].count()
x['Outcome'].tolist()
dados['total'] = 1
x = dados.groupby('Outcome', as_index=False)['total'].count()
labels = x['Outcome'].tolist()
sizes = x['total'].tolist()
plt.figure(figsize=(15, 5))
plt.title('Outcome')
plt.pie(sizes, labels=labels, autopct='%1.0f%%', shadow=False, startangle=15)
plt.legend(labels, loc='best')
plt.axis('equal')

dados['total'] = 1
dados.groupby(by='Outcome')['total'].count().plot.bar()
plt.title('Outcome')

temp = dados.groupby('Outcome', as_index=False)['total'].count()
temp.rename(columns={'Outcome': 'Quantidade'}, inplace=True)
temp['Percentual'] = temp['total'] / dados.shape[0] * 100
temp
plt.style.use('ggplot')
(fig, ax) = plt.subplots(figsize=(20, 10))
ax.set_facecolor('#fafafa')
plt.ylabel('Variaveis')
plt.title('Visão Geral')
ax = sns.boxplot(data=dados, orient='h', palette='Set2')
target = dados['Outcome']
target
explicativas = dados.drop('Outcome', axis=1)
explicativas
(x_treino, x_teste, y_treino, y_teste) = train_test_split(explicativas, target, test_size=0.3, random_state=42)
modelo_log = LogisticRegression(random_state=42, max_iter=400)