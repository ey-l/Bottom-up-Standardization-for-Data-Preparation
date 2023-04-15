import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('data/input/digit-recognizer/train.csv')
df.head()
x = df.drop(['label'], axis=1)
x.head()
y = df['label']
y
a = x.iloc[0:1,]
print('Length of columns :', len(a.columns))
print('Square root for length of a.columns', math.sqrt(len(a.columns)))
b = np.array(a)
ab = np.reshape(b, (28, 28))
plt.matshow(ab)

data = []
for i in range(2, 10):
    aa = x.iloc[i - 1:i,]
    bb = np.array(aa)
    aabb = np.reshape(bb, (28, 28))
    data.append(aabb)
(fig, ax) = plt.subplots(2, 4, figsize=(10, 5))
for i in range(0, 2):
    for j in range(0, 4):
        if i == 0:
            ax[i, j].matshow(data[j])
        elif i == 1:
            ax[i, j].matshow(data[j + 4])
count = pd.DataFrame(df['label'].value_counts()).reset_index()
counts = count.rename(columns={'index': 'label', 'label': 'count'})
counts
plt.figure(figsize=(14, 8))
ax = plt.axes()
ax.set_axisbelow(True)
ax.spines['bottom'].set_color('#000000')
ax.spines['top'].set_color('#000000')
ax.spines['right'].set_color('#000000')
ax.spines['left'].set_color('#000000')
ax.spines['right'].set_linewidth(3)
ax.spines['left'].set_lw(3)
ax.spines['bottom'].set_lw(3)
ax.spines['top'].set_lw(3)
plt.bar(counts['label'], counts['count'], color='#ff5100', label='Salary_1', edgecolor='black')
plt.xlabel('Labels', labelpad=20, loc='center', color='#000000', weight='bold', fontsize=18, fontdict={'family': 'cursive'})
plt.ylabel('Count', labelpad=20, loc='center', color='#000000', weight='bold', fontsize=18, fontdict={'family': 'cursive'})
plt.title('Bar chart', loc='center', color='#000000', weight='bold', fontsize=30, fontdict={'family': 'cursive'}, pad=20, fontstyle='italic')
plt.grid(visible=True, axis='both', which='both', color='#a1a1a1', linestyle='-', linewidth=1, mec='#00cefc')
plt.xticks(ticks=counts['label'], fontsize=15, weight='bold')
plt.yticks(fontsize=15, weight='bold')

(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)
cross_val_score(RandomForestClassifier(), x, y, cv=5)
best_model = RandomForestClassifier(n_estimators=100, criterion='entropy')