"""
Author: Dorottya Demszky (ddemszky@stanford.edu)

Lookup tweets by their ID.
"""

import tweepy
from requests.exceptions import Timeout, ConnectionError
import ssl
import os
import sys
import json
import codecs
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing

# Consumer keys and access tokens, used for OAuth
consumer_keys = []  # Add your list of keys
consumer_secrets = []  # Add your list of secrets
access_tokens = []  # Add your list of access tokens
access_token_secrets = []  # Add your list of token secrets

events = open('all_events/event_names.txt', 'r').read().splitlines()


def get_user(d, user):
    d['user_screen_name'] = user.screen_name
    d['user_id'] = user.id
    d['user_location'] = user.location
    d['user_description'] = user.description
    d['followers_count'] = user.followers_count
    d['friends_count'] = user.friends_count
    d['user_created_at'] = user.created_at.isoformat()
    d['user_statuses_count'] = user.statuses_count
    d['user_favourites_count'] = user.favourites_count
    return d

def get_dict(status):
    d = {}
    d['created_at'] = status.created_at.isoformat()
    d['id'] = status.id
    d = get_user(d, status.user)
    d['geo'] = status.geo
    d['coordinates'] = status.coordinates
    d['text'] = status.text
    d['entities'] = status.entities
    d['in_reply_to_status_id'] = status.in_reply_to_status_id
    d['in_reply_to_user_id'] = status.in_reply_to_user_id
    d['in_reply_to_screen_name'] = status.in_reply_to_screen_name
    d['is_quote_status'] = status.is_quote_status
    d['retweet_count'] = status.retweet_count
    d['favorite_count'] = status.favorite_count
    return d

def get_tweets(oauth):
    # OAuth process, using the keys and tokens
    auth = tweepy.OAuthHandler(consumer_keys[oauth], consumer_secrets[oauth])
    auth.set_access_token(access_tokens[oauth], access_token_secrets[oauth])

    # Creation of the actual interface, using authentication
    api = tweepy.API(auth, retry_count=3, retry_delay=5, retry_errors=set([104, 401, 404, 500, 503]), timeout=2000,
                     wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    for e in events:
        folder = "" # folder to save the tweets in
        try:
            os.mkdir(folder)
        except:
            continue
        print(e)
        try:
            data = pd.read_csv('all_events/'+e+'/' + e + '.csv', usecols=['id'], sep='\t')
        except:
            continue
        ids = [int(i) for i in data[~data['id'].isnull()]['id']]
        print(e, ids)
        lookup_tweets = []
        for i in range(0, len(ids), 100):
            try:
                tweets = api.statuses_lookup(ids[i:i + 100])
            except:
                continue
            with codecs.open(folder+'/tweets.json', 'a', encoding='utf-8') as f:
                for t in tweets:
                    if not t:
                        continue
                    new_dict = {'id': t.id}
                    try:
                        new_dict['retweeted_id'] = t.retweeted_status.id
                        lookup_tweets.append(t.retweeted_status.id)
                    except:
                        pass
                    f.write(json.dumps(new_dict) + '\n')
        for i in range(0, len(lookup_tweets), 100):
            tweets = api.statuses_lookup(lookup_tweets[i:i + 100])
            with codecs.open(folder+'/more_data.json', 'a', encoding='utf-8') as f:
                for t in tweets:
                    if not t:
                        continue
                    f.write(json.dumps(get_dict(t)) + '\n')


Parallel(n_jobs=len(consumer_keys)-1)(delayed(get_tweets)(i) for i in range(1, len(consumer_keys)))
