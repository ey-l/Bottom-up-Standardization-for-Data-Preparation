import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
diabets_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabets_data.head(3)
diabets_data.info()
diabets_data.describe()
diabets_data['Outcome'].value_counts()

def get_clf_eval(y_test, pred=None, pred_proba=None):
    """
    Accuracy, Precision, Recall 
    """
    eval_dict = {}
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('confusion matrix')
    print(confusion)
    print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f},F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
    eval_dict['Accuracy'] = accuracy
    eval_dict['Precision'] = precision
    eval_dict['Recall'] = recall
    eval_dict['F1'] = f1
    eval_dict['ROC AUC'] = roc_auc
    return eval_dict

def precision_recall_curve_plot(y_test=None, pred_proba_c1=None):
    """
    threshold ndarray와 이 threslhold에 따른 정밀도, 재현율 추출 후 시각화 
    """
    (precisions, recalls, thresholds) = precision_recall_curve(y_test, pred_proba_c1)
    pass
    threshold_boundary = thresholds.shape[0]
    pass
    pass
    pass
    pass
    pass
    pass
    pass
X = diabets_data.iloc[:, :-1]
y = diabets_data.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=2021)
lr_clf = LogisticRegression()