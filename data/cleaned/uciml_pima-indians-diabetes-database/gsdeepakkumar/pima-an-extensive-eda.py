import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import itertools
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score, make_scorer, f1_score, auc
from sklearn.tree import export_graphviz

kaggle = 1
if kaggle == 0:
    diab = pd.read_csv('diabetes.csv')
else:
    diab = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print('Number of rows:{} and Number of column:{}'.format(diab.shape[0], diab.shape[1]))
diab.info()
diab.describe()
plt.figure(figsize=(8, 8))
ax = sns.countplot(diab['Outcome'])
ax.set_xlabel('Outcome')
ax.set_ylabel('Count')
ax.set_title('Countplot of Predictor Variable')
count_non_diab = len(diab[diab['Outcome'] == 0])
count_diab = len(diab[diab['Outcome'] == 1])
print('The dataset has {} % of non-diabetic cases and {} % of diabetic cases'.format(round(count_non_diab / len(diab['Outcome']) * 100, 2), round(count_diab / len(diab['Outcome']) * 100, 2)))
diab.isnull().values.any()

def plot_variables(variable):
    (f, axes) = plt.subplots(3, 1, figsize=(10, 10))
    sns.distplot(diab[variable][diab['Outcome'] == 1], ax=axes[0], color='green')
    axes[0].set_title('Distribution of {}-Diabetic Outcome'.format(variable))
    sns.distplot(diab[variable][diab['Outcome'] == 0], ax=axes[1], color='red')
    axes[1].set_title('Distribution of {}-Non Diabetic Outcome'.format(variable))
    sns.boxplot(x=diab['Outcome'], y=diab[variable], ax=axes[2], palette='Set3')
    axes[2].set_title('Boxplot of {}'.format(variable))
    f.tight_layout()
plot_variables('Age')
plot_variables('Pregnancies')
plot_variables('Glucose')
plot_variables('BloodPressure')
plot_variables('SkinThickness')
plot_variables('Insulin')
plot_variables('BMI')
plot_variables('DiabetesPedigreeFunction')
mut_vari = ['BloodPressure', 'Glucose', 'SkinThickness', 'BMI']
for i in mut_vari:
    print('Imputing column {} with median value {} \n'.format(i, diab[i].median()))
    diab[i] = diab[i].replace(0, value=diab[i].median())
X = diab.drop('Outcome', axis=1)
y = diab['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=100)
X_train.shape
X_test.shape

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for (i, j) in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
prediction = {}

def print_score(m):
    res = {'F1 Score for Train:': f1_score(m.predict(X_train), y_train), 'F1 Score for Test:': f1_score(m.predict(X_test), y_test), 'Accuracy Score for Train:': m.score(X_train, y_train), 'Accuracy Score for Test:': m.score(X_test, y_test)}
    if hasattr(m, 'oob_score_'):
        res.append(m.oob_score_)
    print(res)
rfm = RandomForestClassifier(random_state=100, n_jobs=-1, n_estimators=40, min_samples_leaf=5, max_features=0.5)