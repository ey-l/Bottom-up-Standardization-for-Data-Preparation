import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
pd.options.display.max_colwidth = 200
over_140_chars = train[train['text'].str.len() > 140]
over_140_chars['text']
train['location'].value_counts().tail(20)