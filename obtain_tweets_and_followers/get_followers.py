"""
Author: Dorottya Demszky (ddemszky@stanford.edu)

This script saves the follower ids for a list of screen names. 
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

screen_names = []

party = sys.argv[2]

# List of screen names to save followers for
with open('all_'+party+'.txt', 'r') as f:
    screen_names = f.read().splitlines()


for name in screen_names:
    cursor = tweepy.Cursor(api.followers_ids, name).pages()

    i = 0
    if os.path.exists(name+'.txt'):
        continue

    with open(name+'.txt', 'w') as f:
        while True:
            try:
                page = cursor.next()
                print cursor.next_cursor
                f.write(',')
                f.write(','.join(str(x) for x in page))
                print name + ' ' + str(i) + ' ' + str(oauth)
                i += 1
            except (Timeout, ssl.SSLError, ConnectionError), e:
                print e
                pass
            except StopIteration:
                break
            except Exception, e:
                print e
                pass
