import numpy as np


def get_vectors(embeddings, words):
    """
    Input:
        embeddings: a word 
        words: a list of words
    Output: 
        X: a matrix where the rows are the embeddings corresponding to the rows on the list
        
    """
    m = len(words)
    X = np.zeros((1, 300))
    for word in words:
        eng_emb = embeddings[word]
        X = np.row_stack((X, eng_emb))
    X = X[1:,:]
    return X
