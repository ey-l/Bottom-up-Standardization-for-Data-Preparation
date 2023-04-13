import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import csv
from collections import defaultdict
INFILE = 'data/input/spaceship-titanic/train.csv'

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def string_to_float(s, default=0):
    try:
        f = float(s)
        return f
    except ValueError:
        return float(default)
train_data = defaultdict(list)
with open(INFILE) as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        for (k, v) in row.items():
            train_data[k].append(v)
train_data.pop('PassengerId', None)
for elem in train_data['Cabin']:
    try:
        (a, b, c) = elem.split('/')
    except:
        (a, b, c) = ['unknown'] * 3
    train_data['Cabin_a'].append(a)
    train_data['Cabin_b'].append(b)
    train_data['Cabin_c'].append(c)
for elem in train_data['Name']:
    try:
        (fname, lname) = elem.split()
    except:
        (fname, lname) = ['unknown'] * 2
    train_data['FirstName'].append(fname)
    train_data['LastName'].append(lname)
train_data.pop('Cabin', None)
train_data.pop('Name', None)
print(' '.join(train_data.keys()))

def is_feature_categorical(arr):
    arr = np.array(arr)
    unique_val = np.unique(arr)
    if not is_float(unique_val[1]) or len(unique_val) <= 15:
        return True
    else:
        return False
is_categorical = {k: is_feature_categorical(v) for (k, v) in train_data.items()}
for (k, v) in train_data.items():
    train_data[k] = np.array([string_to_float(x, default=-1) for x in v]) if not is_categorical[k] else np.array(v)
train_data = dict(train_data)

def info(data, target):
    print('Features:')
    for feat in data.keys():
        if feat == target:
            continue
        print('  {:20} => {:15}'.format(feat, 'Categorical' if is_categorical[feat] else 'Numerical'), end='')
        if is_categorical[feat]:
            (uniq, c) = np.unique(data[feat], return_counts=True)
            print('Cardinality: {:5} Mode: {:12}  Modecount: {:5}'.format(len(uniq), data[feat][c.argmax()], c.max()))
        else:
            arr = data[feat]
            print('Min: {}  Max: {}  Mean: {:5.1f}'.format(arr.min(), arr.max(), arr.mean()))
    print('Target: ', target)
    if is_categorical[target]:
        print('  Classification task. Classifing to {} classes. Classes are:\n  * '.format(len(np.unique(data[target]))), end='')
        print('\n  * '.join(np.unique(data[target])))
    else:
        print('  Regression task. Target: {}'.format(target))
target_col = 'Transported'
info(train_data, target_col)
for (k, v) in train_data.items():
    print('{:20} length: {}  Unique count: {}'.format(k, len(v), len(np.unique(v))))

def is_pure(data):
    return len(np.unique(data)) == 1

def classify(data):
    (u, c) = np.unique(data, return_counts=True)
    return u[c.argmax()]

def find_split_values(data, target):
    split_vals = {}
    for (col, arr) in data.items():
        if col == target:
            continue
        if is_categorical[col]:
            continue
        values = np.unique(arr)
        split_vals[col] = np.array([(x1 + x2) / 2 for (x1, x2) in zip(values[:-1], values[1:])])
    return split_vals

def entropy(data):
    (_, counts) = np.unique(data, return_counts=True)
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()

def overall_entropy(lhs, rhs):
    tot = lhs.shape[0] + rhs.shape[0]
    p_lhs = lhs.shape[0] / tot
    p_rhs = rhs.shape[0] / tot
    return p_lhs * entropy(lhs) + p_rhs * entropy(rhs)

def stddev(a):
    return a.var() if a.shape[0] != 0 else 0.0

def weighted_mean_squared_error(lhs, rhs):
    tot = lhs.shape[0] + rhs.shape[0]
    p_lhs = lhs.shape[0] / tot
    p_rhs = rhs.shape[0] / tot
    return p_lhs * stddev(lhs) + p_rhs * stddev(rhs)

def find_best_split(data, potential_splits, target):
    criterion = overall_entropy if is_categorical[target] else weighted_mean_squared_error
    best_overall_entropy = 999999999999999999999
    for (col, arr) in data.items():
        if col == target:
            continue
        if is_categorical[col]:
            iter = np.unique(data[col])
        else:
            iter = potential_splits[col]
        for value in iter:
            if is_categorical[col]:
                l_slice = arr == value
                r_slice = arr != value
            else:
                l_slice = arr <= value
                r_slice = arr > value
            d_left = data[target][l_slice]
            d_right = data[target][r_slice]
            if d_left.shape[0] == 0 or d_right.shape[0] == 0:
                continue
            c_entropy = criterion(d_left, d_right)
            if c_entropy <= best_overall_entropy:
                best_overall_entropy = c_entropy
                best_split_column = col
                best_split_value = value
    return (best_split_column, best_split_value)

def split_data(data, column, value):
    if is_categorical[column]:
        l_slice = data[column] == value
        r_slice = data[column] != value
    else:
        l_slice = data[column] <= value
        r_slice = data[column] > value
    l_data = {k: v[l_slice] for (k, v) in data.items()}
    r_data = {k: v[r_slice] for (k, v) in data.items()}
    return (l_data, r_data)
from collections import namedtuple
Decision = namedtuple('Decision', ['column', 'value'])

class DecisionTree(object):

    def __init__(self, data, target, decision=None):
        self.data = data
        self.target = target
        self.decision = decision
        self.left = None
        self.right = None

    def build_tree(self, depth=0, leaf_size_limit=6, max_depth=100):
        tgt_col = self.data[self.target]
        if is_pure(tgt_col):
            return classify(tgt_col) if is_categorical[self.target] else tgt_col.mean()
        if tgt_col.shape[0] < leaf_size_limit:
            return classify(tgt_col) if is_categorical[self.target] else tgt_col.mean()
        if depth > max_depth:
            return classify(tgt_col) if is_categorical[self.target] else tgt_col.mean()
        sv = find_split_values(self.data, self.target)
        self.decision = Decision(*find_best_split(self.data, sv, self.target))
        (l_data, r_data) = split_data(self.data, *self.decision)
        self.left = DecisionTree(l_data, self.target)
        self.right = DecisionTree(r_data, self.target)
        self.left.build_tree(depth=depth + 1, leaf_size_limit=leaf_size_limit, max_depth=max_depth)
        self.right.build_tree(depth=depth + 1, leaf_size_limit=leaf_size_limit, max_depth=max_depth)

    def predict(self, sample):
        tgt_col = self.data[self.target]
        if self.decision is None or is_pure(tgt_col):
            return classify(tgt_col) if is_categorical[self.target] else tgt_col.mean()
        (col, val) = self.decision
        if is_categorical[col]:
            return self.left.predict(sample) if sample[col] == val else self.right.predict(sample)
        else:
            return self.left.predict(sample) if sample[col] <= val else self.right.predict(sample)
split_ratio = 0.1
n_samples = train_data['Transported'].shape[0]
n_train = int((1 - split_ratio) * n_samples)
n_valid = n_samples - n_train
new_train_data = {}
validation_data = {}
for (k, v) in train_data.items():
    new_train_data[k] = v[:n_train]
    validation_data[k] = v[n_train:]
do_not_run = True
for max_d in [20, 22, 24]:
    if do_not_run:
        continue
    my_dt = DecisionTree(new_train_data, target_col)
    my_dt.build_tree(leaf_size_limit=53, max_depth=max_d)
    y_pred = []
    y_real = []
    for i in range(n_valid):
        sample = {k: v[i] for (k, v) in validation_data.items()}
        y_pred.append(my_dt.predict(sample))
        y_real.append(sample[target_col])
    accuracy_array = np.array(y_pred) == np.array(y_real)
    print('Validation accuracy (max_depth={:2d}): {:5.5f} '.format(max_d, accuracy_array.mean()))

class RandomForest(object):

    def __init__(self, n_trees=10):
        self.n_trees = n_trees
        self._trees = []

    def fit(self, data, target, leaf_size_limit=6, max_depth=100):
        self.data = data
        self.target = target
        n_total_samples = data[target].shape[0]
        for i in range(self.n_trees):
            idx = np.unique(np.random.randint(n_total_samples, size=n_total_samples))
            slicer = np.eye(n_total_samples)[idx].sum(axis=0).astype(bool)
            this_tree_data = {k: v[slicer] for (k, v) in data.items()}
            dt = DecisionTree(this_tree_data, target)
            print('Building tree: {:4d} / {:4d}.'.format(i + 1, self.n_trees), end='')
            dt.build_tree(leaf_size_limit=leaf_size_limit, max_depth=max_depth)
            print('', end='\r')
            self._trees.append(dt)
        print('Done!')

    def predict(self, sample):
        arr = np.array([dt.predict(sample) for dt in self._trees])
        if is_categorical[self.target]:
            return classify(arr)
        else:
            return arr.mean()
rf = RandomForest(n_trees=3)