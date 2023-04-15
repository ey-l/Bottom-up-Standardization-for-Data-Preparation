import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
simple_train = ['call you tonight', 'Call me a cab', 'Please call me... PLEASE!']
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()