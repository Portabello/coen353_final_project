import os
import numpy as np
import pandas as pd
import pickle
import tweepy
from datetime import datetime
import re
import time


twitter_keys = {
    'consumer_key': os.environ.get('consumer_key', None),
    'consumer_secret': os.environ.get('consumer_secret', None),
    'access_token_key': os.environ.get('access_token_key', None),
    'access_token_secret': os.environ.get('access_token_secret', None)
}

with open('trained_twitter_model.pickle', 'rb') as read_file:
    xgb_model = pickle.load(read_file)

auth.set_access_token(twitter_keys['access_token_key'], twitter_keys['access_token_secret'])
auth = tweepy.OAuthHandler(twitter_keys['consumer_key'], twitter_keys['consumer_secret'])

api = tweepy.API(auth)


def get_features(screen_name):

    user = api.get_user(screen_name)
    verified = user.verified
    geo_enabled = user.geo_enabled
    followers_count = user.followers_count
    friends_count = user.friends_count
    average_tweets_per_day = np.round(statuses_count / account_age_days, 3)

    account_features = [verified, geo_enabled, followers_count, friends_count,average_tweets_per_day]
    return account_features if len(account_features) == 14 else f'User not found'


def bot_detection(twitter_handle):

    user_features = get_features(twitter_handle)

    if user_features == 'User not found':
        return 'User not found'

    else:
        features = ['verified', 'geo_enabled', 'followers_count', 'friends_count','average_tweets_per_day']

        user_df = pd.DataFrame(np.matrix(user_features), columns=features)

        prediction = xgb_model.predict(user_df)[0]

        return "Bot" if prediction == 1 else "Not a bot"


def bot_accuracy(twitter_handle):
    user_features = get_features(twitter_handle)

    user = np.matrix(user_features)
    proba = np.round(xgb_model.predict_proba(user)[:, 1][0]*100, 2)
    return proba
