import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    if not docs:
        return np.array([], dtype=float)

    N = len(docs)
    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = doc_lengths.mean() if N > 0 else 0.0

    # Term frequencies per document
    doc_counters = [Counter(doc) for doc in docs]

    # Document frequency for each term in corpus
    df = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1

    scores = np.zeros(N, dtype=float)

    for term in query_tokens:
        # If term never appears, idf = 0
        term_df = df.get(term, 0)
        if term_df == 0:
            continue

        idf = math.log((N - term_df + 0.5) / (term_df + 0.5) + 1)

        tf = np.array([counter.get(term, 0) for counter in doc_counters], dtype=float)

        denom = tf + k1 * (1 - b + b * doc_lengths / avgdl)
        term_score = idf * (tf * (k1 + 1)) / denom

        scores += term_score

    return scores