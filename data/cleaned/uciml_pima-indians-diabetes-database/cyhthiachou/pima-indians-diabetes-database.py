import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe()

def show_hist(df):
    return df.hist(figsize=(20, 20))
show_hist(df)
X_full = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
y = X_full.Outcome
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = X_full[features].copy()
(x_temp, x_test, y_temp, y_test) = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
(x_train, x_valid, y_train, y_valid) = train_test_split(x_temp, y_temp, train_size=0.75, test_size=0.25, random_state=0)

def confusion_matrix(pred, y):
    (TP, FP, AP, TN, FN, AN) = (0, 0, 0, 0, 0, 0)
    pred = list(pred)
    y = list(y)
    for i in range(len(pred)):
        if y[i] == 1:
            AP += 1
            if pred[i] == 1:
                TP += 1
            elif pred[i] == 0:
                FN += 1
        elif y[i] == 0:
            AN += 1
            if pred[i] == 1:
                FP += 1
            elif pred[i] == 0:
                TN += 1
    recall_rate = TP / AP
    specificity_rate = TN / AN
    accuracy_rate = (TP + TN) / len(y)
    misclassification_rate = (FP + FN) / len(y)
    precision = TP / (TP + FP)
    f1_score = 2 * (recall_rate * precision) / (recall_rate + precision)
    print('recall_rate is {}, specificity_rate is {}, accuracy_rate is {}, misclassification_rate is {}, f1_score is {}'.format(round(recall_rate, 2), round(specificity_rate, 2), round(accuracy_rate, 2), round(misclassification_rate, 2), round(f1_score, 2)))
    return (TP, FP, TN, FN, AP, AN, recall_rate, specificity_rate, accuracy_rate, misclassification_rate, f1_score)

def score_dataset(x_train, y_train, x_valid, y_valid, model):
    scaler = StandardScaler()
    scaled_x_train = scaler.fit_transform(x_train)