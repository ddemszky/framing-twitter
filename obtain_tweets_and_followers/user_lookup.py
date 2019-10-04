"""
Author: Dorottya Demszky (ddemszky@stanford.edu)

Lookup users based on their IDs (see if they are deactivated).
"""


import tweepy
from requests.exceptions import Timeout, ConnectionError
import ssl
import os
import sys


oauth = int(sys.argv[1])
# Consumer keys and access tokens, used for OAuth
consumer_keys = []  # Add your list of keys
consumer_secrets = []  # Add your list of secrets
access_tokens = []  # Add your list of access tokens
access_token_secrets = []  # Add your list of token secrets

# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_keys[oauth], consumer_secrets[oauth])
auth.set_access_token(access_tokens[oauth], access_token_secrets[oauth])

# Creation of the actual interface, using authentication
api = tweepy.API(auth, retry_count=3, retry_delay=5, retry_errors=set([104, 401, 404, 500, 503]), timeout=2000, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

users = []

with open('users.txt', 'r') as f:
    users = f.read().splitlines()

total = len(users)


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

