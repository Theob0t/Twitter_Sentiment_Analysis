from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
import gensim
TaggedDocument = gensim.models.doc2vec.TaggedDocument
import GetOldTweets3 as got
import numpy as np
from get_word2vec import buildWordVector
from sklearn.preprocessing import scale

def tokenize(tweet):
    tweet = tweet.lower()
    tweet = ' '.join([ t for t in tweet.split() if not t.startswith(('@','#','http')) ])
    tokens = tokenizer.tokenize(tweet)
    return tokens


def scrapping_twitter(username='realDonaldTrump'):
    tweetCriteria = got.manager.TweetCriteria().setUsername(username)\
                                               .setMaxTweets(5)
    
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    for t in range(len(tweets)):
        try:
            if len(tweets[t].text)!=0:
                tweet = got.manager.TweetManager.getTweets(tweetCriteria)[t]
                return tweet.text
            else:
                continue
        except: 
            return 'ERROR - NO TWEET FOUND'


def tweet_pipeline(tweet):

    '''
    Transformation Pipeline of the input tweet.
    STEP:
    - Tokenize the tweet => ['i', 'love', 'you']
    - Convert to LabeledSentence gensim object
    - Build tweet vectorization (word2vec)
    - Scale (zero mean and unit standard deviation)
    
    input:
    raw tweet, string
    output:
    vectorized tweet, np.array (200,)
    '''

    trump_tweet_tokenized = tokenize(tweet)
    
    trump_tweet_labelized = TaggedDocument(trump_tweet_tokenized,'TEST')
    
    tweet_vec = np.concatenate(buildWordVector(trump_tweet_labelized.words, 200))
    
    tweet = [scale(tweet_vec)]
    
    return tweet
