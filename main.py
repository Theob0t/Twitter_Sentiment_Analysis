import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sentiment_analysis import sentiment_analysis
from get_word2vec import buildWordVector
from get_tweet import tokenize, scrapping_twitter, tweet_pipeline
from tensorflow.keras.models import load_model
import numpy as np
from termcolor import colored




def tweet_sentimental_analysis(username='realDonaldTrump'):
    # load a saved model
    saved_model = load_model('models/best_model.h5')
    tweet = scrapping_twitter(username)
    prediction = round(sentiment_analysis(tweet, saved_model)[0][0],2)
    if prediction == -1:
        print(colored('    ERROR - NO TWEET FOUND FOR ', attrs=['reverse']) + colored('"',attrs=['reverse']) + colored(username,attrs=['reverse']) + colored('"',attrs=['reverse']) + colored('    ',attrs=['reverse']))
        return
    elif prediction < 0.4:
        status = 'Negatif'
        color = 'red'
    elif 0.4 <= prediction <= 0.6:
        status = 'Neutral'
        color = 'white'
    else:
        status = 'Positif'
        color = 'green'
    print(colored('Last Tweet :', 'cyan'), colored(tweet, attrs=['reverse']))
    print(colored('Sentimental Score :', 'yellow'), prediction)
    print(colored('Current Mood :', color), colored(status, color=color))
    return 

if __name__ == '__main__':
    
    print()
    print(colored('===========TWEET SENTIMENT ANALYSIS===========', 'blue'))
    username = input('Enter Twitter username : ')
    print('--------------' + colored('Current Mood Checker', 'red') + '--------------')
    tweet_sentimental_analysis(username)
    print(colored('================================================','blue'))
    print()