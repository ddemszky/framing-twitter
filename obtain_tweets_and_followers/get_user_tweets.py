"""
Author: Dorottya Demszky (ddemszky@stanford.edu)

Download tweets for a particular user.
"""

import tweepy
from requests.exceptions import Timeout, ConnectionError
import ssl
import sys
import codecs
import csv
import os.path
import iso8601
import time


oauth = int(sys.argv[1])
event = sys.argv[2]  # name of the event
filename = sys.argv[3] # dem_users or rep_users
folder = event
path = "" # your filepath

# Consumer keys and access tokens, used for OAuth
consumer_keys = []  # Add your list of keys
consumer_secrets = []  # Add your list of secrets
access_tokens = []  # Add your list of access tokens
access_token_secrets = []  # Add your list of token secrets


# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_keys[oauth], consumer_secrets[oauth])
auth.set_access_token(access_tokens[oauth], access_token_secrets[oauth])

# Creation of the actual interface, using authentication
api = tweepy.API(auth, retry_count=0, retry_delay=5, retry_errors=set([104, 401, 404, 500, 503]), timeout=100, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

users = []

with open(folder + '/user_tweets/'+filename+'.txt', 'r') as f:
    users = f.read().splitlines()

total = len(users)


for i, u in enumerate(users):
    if i % 1000 == 0:
        print(i)
    fname = path +folder+'/user_tweets/%s_tweets.csv' % u
    if os.path.isfile(fname):
        print(fname + ' exists')
        continue
    try:
        tweets = api.user_timeline(screen_name=u, count=200)
        outtweets = [[u, tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in tweets]
        with codecs.open(fname, 'wb', encoding='utf-8') as f:
            print('created ' + fname)
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["user_screen_name", "id", "created_at", "text"])
            writer.writerows(outtweets)
    except tweepy.TweepError as e:
        with codecs.open(fname, 'wb', encoding='utf-8') as f:
            f.write('None.')
        print(e)
        pass


'''
i = 0
with open('active_ids.txt', 'w') as f:
    while i < (total / 100) + 1:
        try:
            f.write('\n'.join([str(u.id) for u in api.lookup_users(user_ids=users[i*100:min((i+1)*100, total)])]))
            f.write('\n')
            print 'getting users batch:', i
            i += 1
        except tweepy.TweepError as e:
            print e
            pass

'''