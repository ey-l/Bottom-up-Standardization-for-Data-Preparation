import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
pd.options.display.max_colwidth = 200
over_140_chars = _input1[_input1['text'].str.len() > 140]
over_140_chars['text']
_input1['location'].value_counts().tail(20)