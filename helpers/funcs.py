#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import re
import nltk
import string
import json
import pandas as pd

sno = nltk.stem.SnowballStemmer('english')
punct_chars = list((set(string.punctuation) | {'’', '‘', '–', '—', '~', '|', '“', '”', '…', "'", "`", '_'}) - set(['#']))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))

config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
TWEET_DIR = config['TWEET_DIR']
NUM_CLUSTERS = config['NUM_CLUSTERS']


stopwords = set(open(INPUT_DIR + 'stopwords.txt', 'r').read().splitlines())
event_stopwords = json.load(open(INPUT_DIR + "event_stopwords.json","r"))

def clean_text(text, keep_stopwords=True, event=None, stem=True):
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
    words = text.split()
    if not keep_stopwords:
        words = [w for w in words if w not in stop]
    if stem:
        words = [sno.stem(w) for w in words]
    return words

def filter_retweets(data):
    # make sure to not reset the indices!!! (that would cause an issue with the other filtering methods)
    return data[~data['remove'] & ~data['isRT']]

def split_party(data):
    part_tweets = data[~data['dem_follows'].isnull() & ~data['rep_follows'].isnull() & (data['dem_follows'] != data['rep_follows'])]
    return part_tweets[part_tweets['dem_follows'] > part_tweets['rep_follows']], part_tweets[part_tweets['dem_follows'] < part_tweets['rep_follows']]

def method_name(cluster_method):
    if cluster_method:
        return '_' + cluster_method
    else:
        return ''

def get_assigned_indices_relative(topics):
    threshold = .906  # approx. 75% of all distance ratios for 8 topics
    topics['ratio'] = topics['cosine_0'] / topics['cosine_1']
    topics = topics[topics['ratio'] < threshold]
    return topics.index.astype(int), topics

def get_assigned_indices_absolute(topics):
    threshold = .62  # approx. 75% of distances to closest centroid for 6 topics
    topics = topics[topics['cosine_0'] < threshold]
    return topics.index.astype(int), topics

def get_cluster_assignments(event, data, cluster_method):
    '''
        :param
            event: name of the event
            data: dataframe of tweets
            method: "relative": based on the ratio of the cosine distances of the 1st and 2nd closest cluster (we use this in  the paper)
                        "absolute": based on absolute cosine distance of the closest cluster
                        None (default): assign all tweets to the closest cluster
        :return:
    '''
    # load topics
    topics = pd.read_csv(TWEET_DIR + event + '/' + event + '_kmeans_topics_' + str(NUM_CLUSTERS) + '.csv')
    data = data.iloc[topics['indices_in_original'].astype(int)]
    data.reset_index(drop=True, inplace=True)

    # filter clustered tweets
    if cluster_method:
        assigned_indices, topics = get_assigned_indices_relative(topics) if cluster_method == 'relative' else get_assigned_indices_absolute(topics)

        data = data.iloc[assigned_indices]
        data.reset_index(drop=True, inplace=True)
        topics.reset_index(drop=True, inplace=True)

    # assign clusters
    data['topic'] = topics['topic_0'].astype(int)

    return data

def get_buckets(data, timestamp, no_splits, split_by):
    '''Divide tweets into time buckets.'''
    hour = 60 * 60
    day = 24 * hour
    timestamps = data['timestamp'].astype(float)
    buckets = []
    times = []
    start = timestamp
    for i in range(no_splits):
        new_start = start + split_by
        b = data[(timestamps > start) & (timestamps < new_start)]
        start = new_start
        buckets.append(b)
        times.append(split_by * (i + 1) / day)
    return buckets, times

def get_buckets_log(data, timestamp, no_splits, split_by):
    '''Divide tweets into time buckets.'''
    hour = 60 * 60
    day = 24 * hour
    data['timestamp'] = data['timestamp'].astype(float) - timestamp
    data = data[(data['timestamp'] > 0) & (data['timestamp'] < (no_splits * split_by))]
    data['timestamp'] = np.log(data['timestamp'])
    bins = np.linspace(data['timestamp'].min(), data['timestamp'].max(), no_splits, endpoint=False)
    groups = data.groupby(np.digitize(data['timestamp'], bins))
    buckets = []
    times = []
    for i, group in groups:
        print(i, len(group))
        t = np.exp((group['timestamp'].min() + group['timestamp'].max()) / 2) / day # turn it back into seconds, then convert to days
        print(t)
        times.append(t)
        buckets.append(group)
    return buckets, times