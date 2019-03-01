import numpy as np
import re
import nltk
import string
import json

sno = nltk.stem.SnowballStemmer('english')
punct_chars = list((set(string.punctuation) | {'’', '‘', '–', '—', '~', '|', '“', '”', '…', "'", "`", '_'}) - set(['#']))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))

config = json.load(open('../config.json', 'r'))
DATA_DIR = config['DATA_DIR']

stopwords = set(open(DATA_DIR + 'stopwords.txt', 'r').read().splitlines())
event_stopwords = json.load(open(DATA_DIR + "event_stopwords.json","r"))

def clean_text(text, keep_stopwords=True, event=None):
    if not keep_stopwords:
        stop = stopwords | set(event_stopwords[event])
    # lower case
    text = text.lower()
    # eliminate urls
    text = re.sub(r'http\S*|\S*\.com\S*|\S*www\S*', ' ', text)
    # eliminate @mentions
    text = re.sub(r'\s@\S+', ' ', text)
    # substitute all other punctuation with whitespace
    text = replace.sub(' ', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    # stem words
    if not keep_stopwords:
        return [sno.stem(w) for w in text.split() if w not in stop]
    return [sno.stem(w) for w in text.split()]

def filter_retweets(data):
    return data[~data['remove'] & ~data['isRT']]

def split_party(data):
    part_tweets = data[~data['dem_follows'].isnull() & ~data['rep_follows'].isnull() & (data['dem_follows'] != data['rep_follows'])]
    return part_tweets[part_tweets['dem_follows'] > part_tweets['rep_follows']], part_tweets[part_tweets['dem_follows'] < part_tweets['rep_follows']]

def method_name(cluster_method):
    if cluster_method:
        return '_' + cluster_method
    else:
        return ''

def filter_clustered_tweets(event, data, tweet_dir, cluster_method):
    indices = np.load(tweet_dir + event + '/' + event + '_cleaned_and_partisan_indices.npy')  # tweets that have embeddings
    data = data.iloc[indices]
    data.reset_index(drop=True, inplace=True)
    cluster_method = method_name(cluster_method)
    assigned_indices = np.load(
        tweet_dir + event + '/' + event + '_cluster_assigned_embed_indices' + cluster_method + '.npy')
    data = data.iloc[assigned_indices]
    data.reset_index(drop=True, inplace=True)
    return data

def get_clusters(event, tweet_dir, cluster_method, num_clusters):
    cluster_method = method_name(cluster_method)
    return np.load(tweet_dir + event + '/' + event + '_cluster_labels_' + str(num_clusters) + cluster_method + '.npy')

def get_buckets(data, timestamp, no_splits, split_by):
    '''Divide tweets into time buckets.'''
    timestamps = data['timestamp'].astype(float)
    buckets = []
    start = timestamp
    for i in range(no_splits):
        new_start = start + split_by
        b = data[(timestamps > start) & (timestamps < new_start)]
        start = new_start
        buckets.append(b)
    return buckets