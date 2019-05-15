#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import json
import sys
import scipy.sparse as sp
import pandas as pd
from collections import Counter
sys.path.append('..')
from helpers.funcs import *

from calculate_polarization import get_values
config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']
TWEET_DIR = config['TWEET_DIR']
NUM_CLUSTERS = config['NUM_CLUSTERS']

events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()
print(events)

def mutual_information(dem_counts, rep_counts):
    assert(dem_counts.shape[1] == rep_counts.shape[1])
    dem_no = dem_counts.shape[0]
    rep_no = rep_counts.shape[0]
    no_users = dem_no + rep_no
    no_tokens = dem_counts.shape[1]

    # get nonzero values
    dem_nonzero = sp.find(dem_counts)[:2]
    rep_nonzero = sp.find(rep_counts)[:2]

    dem_t_counts = Counter(dem_nonzero[1]) # number of Dem users using each term
    rep_t_counts = Counter(rep_nonzero[1]) # number of Rep users using each term
    dem_t = np.ones(no_tokens)  # add one smoothing
    rep_t = np.ones(no_tokens)
    for k, v in dem_t_counts.items():
        dem_t[k] += v
    for k, v in rep_t_counts.items():
        rep_t[k] += v
    dem_not_t = dem_no - dem_t + 2  # because of add one smoothing
    rep_not_t = rep_no - rep_t + 2
    all_t = dem_t + rep_t
    all_not_t = no_users - all_t + 4

    mi_dem_t = dem_t * np.log2(no_users * (dem_t / (all_t * dem_no)))
    mi_dem_not_t = dem_not_t * np.log2(no_users * (dem_not_t / (all_not_t * dem_no)))
    mi_rep_t = rep_t * np.log2(no_users * (rep_t / (all_t * rep_no)))
    mi_rep_not_t = rep_not_t * np.log2(no_users * (rep_not_t / (all_not_t * rep_no)))
    mi_values = (1 / no_users * (mi_dem_t + mi_dem_not_t + mi_rep_t + mi_rep_not_t)).transpose()[:, np.newaxis]

    return mi_values, mi_values


def get_polarization(event, method = "nofilter", cluster_method = None, between_topic=False):
    '''
    :param event: name of the event (str)
    :param method: "nofilter" (default): use all tweets
                    "noRT": ignore retweets only
                    "clustered": keep only tweets that were assigned to clusters; this is a subset of "cleaned
    :param cluster_method: None, "relative" or "absolute" (see 5_assign_tweets_to_clusters.py); must have relevant files --> relative is used in paper
    :return: tuple: (true value, random value)
    '''
    data = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n', usecols=['text', 'timestamp', 'user_id', 'dem_follows', 'rep_follows', 'remove', 'isRT'])
    if method == "noRT":
        data = filter_retweets(data)
    elif method == 'clustered':
        data = get_cluster_assignments(event, data, cluster_method)
    elif method != "nofilter":
        print("invalid method.")
        return None

    print(event, len(data))
    return get_values(event, data, token_partisanship_measure=mutual_information, leaveout=True,
                      between_topic=between_topic, default_score=0)

def get_polarization_topics(event, cluster_method = None):
    '''
    :param event: name of the event (str)
    :param cluster_method: None, "relative" or "absolute" (see 5_assign_tweets_to_clusters.py); must have relevant files --> relative is used in paper
    :return: tuple: (true value, random value)
    '''
    data = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['user_id', 'text', 'dem_follows', 'rep_follows'])
    data = get_cluster_assignments(event, data, cluster_method)

    print(event, len(data))

    topic_polarization = {}
    for i in range(NUM_CLUSTERS):
        print(i)
        topic_polarization[i] = tuple(get_values(event, data[data['topic'] == i], token_partisanship_measure=mutual_information))

    cluster_method = method_name(cluster_method)
    with open(TWEET_DIR + event + '/' + event + '_mutual_information_topic_polarization' + cluster_method + '.json', 'w') as f:
        f.write(json.dumps(topic_polarization))

if __name__ == "__main__":

    # for overall polarization

    event_polarization = {}
    method = sys.argv[1]
    cluster_method = None if len(sys.argv) < 3 else sys.argv[2]
    for e in events:
        event_polarization[e] = tuple(get_polarization(e, method, cluster_method))

    cluster_method = method_name(cluster_method)
    with open(OUTPUT_DIR + 'mutual_information_' + method + cluster_method + '_leaveout.json', 'w') as f:
        f.write(json.dumps(event_polarization))

    # for between topic polarization
    '''
    between_topic_polarization = {}
    cluster_method = None if len(sys.argv) < 2 else sys.argv[1]
    for e in events:
        between_topic_polarization[e] = tuple(get_polarization(e, "clustered", cluster_method, between_topic=True))

    cluster_method = method_name(cluster_method)
    with open(OUTPUT_DIR + 'mutual_information_between_topic_polarization' + cluster_method + '.json', 'w') as f:
        f.write(json.dumps(between_topic_polarization))
    '''

    # for within topic polarization
    '''
    topic_polarization = {}
    cluster_method = None if len(sys.argv) < 2 else sys.argv[1]
    for e in events:
        get_polarization_topics(e, cluster_method)
    '''
