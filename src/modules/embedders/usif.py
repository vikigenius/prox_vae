#!/usr/bin/env python
import re
import numpy as np
from sklearn.decomposition import TruncatedSVD
from nltk import word_tokenize


class Word2Prob(object):
    """Map words to their probabilities."""
    def __init__(self, count_fn):
        """Initialize a word2prob object.
        Args:
            count_fn: word count file name (one word per line)
        """
        self.prob = {}
        total = 0.0

        for line in open(count_fn):
            k, v = line.split()
            v = int(v)
            k = k.lower()

            self.prob[k] = v
            total += v

        self.prob = {k: (self.prob[k] / total) for k in self.prob}
        self.min_prob = min(self.prob.values())
        self.count = total

    def __getitem__(self, w):
        return self.prob.get(w.lower(), self.min_prob)

    def __contains__(self, w):
        return w.lower() in self.prob

    def __len__(self):
        return len(self.prob)

    def vocab(self):
        return iter(self.prob.keys())


class Word2Vec(object):
    """Map words to their embeddings."""
    def __init__(self, vector_fn):
        """Initialize a word2vec object.
        Args:
            vector_fn: embedding file name (one word per line)
        """
        self.vectors = {}

        for line in open(vector_fn):
            line = line.split()

            # skip first line if needed
            if len(line) == 2:
                continue

            word = line[0]
            embedding = np.array([float(val) for val in line[1:]])
            self.vectors[word] = embedding

    def __getitem__(self, w):
        return self.vectors[w]

    def __contains__(self, w):
        return w in self.vectors


class USIF(object):
    """Embed sentences using unsupervised smoothed inverse frequency."""
    def __init__(self, vec, prob, n=11, m=5):
        """Initialize a sent2vec object.
        Variable names (e.g., alpha, a) all carry over from the paper.
        Args:
            vec: word2vec object
            prob: word2prob object
            n: expected random walk length. This is the avg sentence length, which
                should be estimated from a large representative sample. For STS
                tasks, n ~ 11. n should be a positive integer.
            m: number of common discourse vectors (in practice, no more than 5 needed)
        """
        self.vec = vec
        self.m = m

        if not (isinstance(n, int) and n > 0):
            raise TypeError("n should be a positive integer")

        vocab_size = float(len(prob))
        threshold = 1 - (1 - 1/vocab_size) ** n
        alpha = len([w for w in prob.vocab() if prob[w] > threshold]) / vocab_size
        Z = 0.5 * vocab_size
        self.a = (1 - alpha)/(alpha * Z)

        self.weight = lambda word: (self.a / (0.5 * self.a + prob[word]))

    def _to_vec(self, sentence):
        """Vectorize a given sentence.

        Args:
            sentence: a sentence (string)
        """
        # regex for non-punctuation
        not_punc = re.compile('.*[A-Za-z0-9].*')

        # preprocess a given token
        def preprocess(t):
            t = t.lower().strip("';.:()").strip('"')
            t = 'not' if t == "n't" else t
            return re.split(r'[-]', t)

        tokens = []

        for token in word_tokenize(sentence):
            if not_punc.match(token):
                tokens = tokens + preprocess(token)

        tokens = list(filter(lambda t: t in self.vec, tokens))

        # if no parseable tokens, return a vector of a's
        if tokens == []:
            return np.zeros(300) + self.a
        else:
            v_t = np.array([self.vec[t] for t in tokens])
            v_t = v_t * (1.0 / np.linalg.norm(v_t, axis=0))
            v_t = np.array([self.weight(t) * v_t[i, :] for i, t in enumerate(tokens)])
            return np.mean(v_t, axis=0)

    def embed(self, sentences):
        """Embed a list of sentences.
        Args:
            sentences: a list of sentences (strings)
        """
        vectors = [self._to_vec(s) for s in sentences]

        if self.m == 0:
            return vectors

        def proj(a, b):
            return a.dot(b.transpose()) * b

        svd = TruncatedSVD(n_components=self.m, random_state=0).fit(vectors)

        # remove the weighted projections on the common discourse vectors
        for i in range(self.m):
            lambda_i = (svd.singular_values_[i] ** 2) / (svd.singular_values_ ** 2).sum()
            pc = svd.components_[i]
            vectors = [v_s - lambda_i * proj(v_s, pc) for v_s in vectors]

        return vectors


# Load usif
def get_paranmt_usif(m=5):
    """Return a uSIF embedding model that used pre-trained ParaNMT word vectors."""
    prob = Word2Prob('data/interim/embeddings/paranmt/enwiki_vocab_min200.txt')
    vec = Word2Vec('data/interim/embeddings/paranmt/czeng.txt')
    return USIF(vec, prob, m=m)
