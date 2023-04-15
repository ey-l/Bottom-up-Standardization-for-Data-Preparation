import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

train_data = pd.read_csv('data/input/nlp-getting-started/train.csv')
train_data = train_data.drop(['location', 'keyword'], axis=1)
train_data = train_data.dropna(axis=0)
train_data.head()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline

def train_model(model, data, targets):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', model)])