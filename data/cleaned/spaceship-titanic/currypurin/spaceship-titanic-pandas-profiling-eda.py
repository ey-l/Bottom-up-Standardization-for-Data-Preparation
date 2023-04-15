from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
COMPE_DIR = Path('data/input/spaceship-titanic/')
TARGET_COL = 'Transported'
train = pd.read_csv(COMPE_DIR / 'train.csv')
test = pd.read_csv(COMPE_DIR / 'test.csv')
sample_submission = pd.read_csv(COMPE_DIR / 'sample_submission.csv')
(train.shape, test.shape, sample_submission.shape)


train[TARGET_COL].value_counts()
import pandas_profiling as pdp
pdp.ProfileReport(train)
pdp.ProfileReport(test)