import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    if not documents:
        return np.zeros((0, 0), dtype=float), []

    # tokenize
    tokenized_docs = [doc.lower().split() for doc in documents]

    # build sorted vocabulary
    vocabulary = sorted(set(token for doc in tokenized_docs for token in doc))

    if not vocabulary:
        return np.zeros((len(documents), 0), dtype=float), []

    word_to_idx = {word: i for i, word in enumerate(vocabulary)}

    n_docs = len(documents)
    n_vocab = len(vocabulary)

    # document frequency
    df = Counter()
    for doc in tokenized_docs:
        for word in set(doc):
            df[word] += 1

    # idf
    idf = np.zeros(n_vocab, dtype=float)
    for word, idx in word_to_idx.items():
        idf[idx] = math.log(n_docs / df[word])

    # tf-idf matrix
    tfidf_matrix = np.zeros((n_docs, n_vocab), dtype=float)

    for doc_idx, doc in enumerate(tokenized_docs):
        if not doc:
            continue

        counts = Counter(doc)
        total_terms = len(doc)

        for word, count in counts.items():
            col_idx = word_to_idx[word]
            tf = count / total_terms
            tfidf_matrix[doc_idx, col_idx] = tf * idf[col_idx]

    return tfidf_matrix, vocabulary