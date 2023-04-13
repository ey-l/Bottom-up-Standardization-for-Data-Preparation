import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1 = _input1.drop(['location', 'keyword'], axis=1)
_input1 = _input1.dropna(axis=0)
_input1.head()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline

def train_model(model, data, targets):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', model)])