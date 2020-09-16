from tensorflow.keras.models import load_model
from gensim.models.word2vec import Word2Vec
import pickle
import numpy as np


def buildWordVector(tokens, size):

    tweet_w2v = Word2Vec.load('models/word2vec.model')

    tfidf = pickle.load(open('models/tfidf.pickle','rb'))
    
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
            
    #avoid 0 division
    if count != 0:
        vec /= count
    
    open('models/tfidf.pickle','rb').close()

    return vec