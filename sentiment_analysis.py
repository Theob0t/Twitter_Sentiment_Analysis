from get_word2vec import buildWordVector
from get_tweet import tokenize, scrapping_twitter, tweet_pipeline
import numpy as np

# # load a saved model
# from tensorflow.keras.models import load_model
# saved_model = load_model('best_model.h5')

def sentiment_analysis(tweet, model):
    vect_tweet = tweet_pipeline(tweet)
    if tweet != 'ERROR - NO TWEET FOUND':
        if (vect_tweet[0].ndim == 1):
            vect_tweet[0] = np.array([vect_tweet[0]])
        prediction = model.predict(vect_tweet[0])
        #print('This is the prediction', prediction[0][0])
        return prediction
    
    else:
        prediction = [[-1]]
        return prediction


#sentiment_analysis(scrapping_twitter(), saved_model)