import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_X_y, check_array
path = 'data/input/nlp-getting-started/'
train_df = pd.read_csv(path + 'train.csv')
submit_df = pd.read_csv(path + 'test.csv')
y_train = train_df['target']
y_train.unique()
train_df.head()
vocabulary = []
_ = [vocabulary.extend(x.split()) for (i, x) in enumerate(train_df['text'])]
vocabulary = np.array(vocabulary)
vocab = np.unique(vocabulary)
print('Vocab:', len(vocab))
vectorizer = CountVectorizer(vocabulary=vocab)
word_counts = vectorizer.fit_transform(train_df.text.to_numpy()).toarray()
X_train = pd.DataFrame(word_counts, columns=vocab).to_numpy()

class NaiveBayes:

    def __init__(self, alpha=1.0):
        self.prior = None
        self.word_counts = None
        self.lk_word = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior, word_counts and lk_word
        """
        (x, y) = check_X_y(x, y)
        n = x.shape[0]
        x_per_class = np.array([x[y == c] for c in np.unique(y)])
        self.prior = np.array([len(x_class) / n for x_class in x_per_class])
        self.word_counts = np.array([sub_arr.sum(axis=0) for sub_arr in x_per_class]) + self.alpha
        self.lk_word = self.word_counts / self.word_counts.sum(axis=1).reshape(-1, 1)
        return self

    def _get_class_numerators(self, x):
        """
        Calculate for each class, the likelihood that an entire message conditional
        on the message belonging to a particular class (real or fake)
        """
        (n, m) = (x.shape[0], self.prior.shape[0])
        class_numerators = np.zeros(shape=(n, m))
        for (i, word) in enumerate(x):
            word_exists = word.astype(bool)
            lk_words_present = self.lk_word[:, word_exists] ** word[word_exists]
            lk_message = lk_words_present.prod(axis=1)
            class_numerators[i] = lk_message * self.prior
        return class_numerators

    def _normalized_conditional_probs(self, class_numerators):
        """
        Conditional probabilities = class_numerators / normalize_term
        """
        normalize_term = class_numerators.sum(axis=1).reshape(-1, 1)
        conditional_probs = class_numerators / normalize_term
        assert (conditional_probs.sum(axis=1) - 1 < 0.001).all(), 'rows should sum to 1'
        return conditional_probs

    def predict_proba(self, x):
        """
        Return the probabilities for each class (fake or real)
        """
        class_numerators = self._get_class_numerators(x)
        conditional_probs = self._normalized_conditional_probs(class_numerators)
        return conditional_probs

    def predict(self, x):
        """
        Return the answer with the highest probability (argmax)
        """
        return self.predict_proba(x).argmax(axis=1)