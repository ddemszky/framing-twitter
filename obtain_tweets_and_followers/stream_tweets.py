"""
Author: Dorottya Demszky (ddemszky@stanford.edu)

Stream tweets based on a list of queries.
"""

import tweepy
from requests.exceptions import Timeout, ConnectionError
import ssl
import os
import sys
import codecs
import json


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
api = tweepy.API(auth, retry_count=0, retry_delay=5, retry_errors=set([104, 401, 404, 500, 503]), timeout=2000, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


def get_user(d, user):
    d['user_screen_name'] = user.screen_name
    d['user_id'] = user.id
    d['user_location'] = user.location
    d['user_description'] = user.description
    d['followers_count'] = user.followers_count
    d['friends_count'] = user.friends_count
    d['user_created_at'] = user.created_at.isoformat()
    d['user_statuses_count'] = user.statuses_count
    d['user_listed_count'] = user.listed_count
    d['user_favourites_count'] = user.favourites_count
    return d

def get_dict(status):
    d = {}
    d['created_at'] = status.created_at.isoformat()
    d['id'] = status.id
    d = get_user(d, status.user)
    d['geo'] = status.geo
    d['coordinates'] = status.coordinates
    try:
        d['retweeted_id'] = status.retweeted_status.id
        d['retweeted_text'] = status.retweeted_status.text
        d['retweeted_created_at'] = status.retweeted_status.created_at.isoformat()
        d['retweeted_user_screen_name'] = status.retweeted_status.user.screen_name
        d['retweeted_user_id'] = status.retweeted_status.user.id
    except:
        d['text'] = status.text
        d['truncated'] = status.truncated
        d['entities'] = status.entities
        d['in_reply_to_status_id'] = status.in_reply_to_status_id
        d['in_reply_to_user_id'] = status.in_reply_to_user_id
        d['in_reply_to_screen_name'] = status.in_reply_to_screen_name
    d['is_quote_status'] = status.is_quote_status
    d['retweet_count'] = status.retweet_count
    d['favorite_count'] = status.favorite_count
    return d





# Inherit from the StreamListener object
class MyStreamListener(tweepy.StreamListener):
    # Overload the on_status method
    def on_status(self, status):
        try:

            # Open a text file to save tweets to
            with codecs.open('all_events/thousandoaks.json', 'a', encoding='utf-8') as f:

                # Check if the tweet has coordinates, if so write it to text
                f.write(json.dumps(get_dict(status)) + '\n')
                return True

        # Error handling
        except BaseException as e:
            print("Error on_status: %s" % str(e))

        return True

    # Error handling
    def on_error(self, status):
        print(status)
        return True

    # Timeout handling
    def on_timeout(self):
        return True


#Create a stream object
twitter_stream = tweepy.Stream(auth, MyStreamListener())
twitter_stream.filter(languages=["en"], track=["Thousand Oaks", "thousandoaks", "ThousandOaks", "California shooting"], async=True)
'''
twitter_stream.filter(languages=["en"], track=['Pittsburgh','pittsburgh','#PittsburghSynagogue', \
                             '#Pittsburgh','#PittsburghSynagogue','Bowers',\
                             '#PittsburghStrong','#PittsburghShooting','Tree of Life', \
                             'synagogue','Synagogue','#PittsburghSynagogueShooting', \
                             '#TreeOfLifeSynagogue'], async=True)
'''