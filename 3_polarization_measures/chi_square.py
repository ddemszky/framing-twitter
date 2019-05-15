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

def chi_square(dem_counts, rep_counts, exclude_user_party = None, exclude_user_id = None):
    assert(dem_counts.shape[1] == rep_counts.shape[1])
    dem_no = dem_counts.shape[0]
    rep_no = rep_counts.shape[0]
    if exclude_user_party == 'Dem':
        dem_no -= 1
        excl_user_terms = sp.find(dem_counts[exclude_user_id, :])[1]
    elif exclude_user_party == 'Rep':
        rep_no -= 1
        excl_user_terms = sp.find(rep_counts[exclude_user_id, :])[1]
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
    if exclude_user_party == 'Dem':
        for term_idx in excl_user_terms:
            dem_t[term_idx] -= 1
    elif exclude_user_party == 'Rep':
        for term_idx in excl_user_terms:
            rep_t[term_idx] -= 1
    dem_not_t = dem_no - dem_t + 2  # because of add one smoothing
    rep_not_t = rep_no - rep_t + 2
    all_t = dem_t + rep_t
    all_not_t = no_users - all_t + 4

    chi_enum = no_users * (dem_t * rep_not_t - dem_not_t * rep_t) ** 2
    chi_denom = all_t * all_not_t * (dem_t + dem_not_t) * (rep_t + rep_not_t)
    chi_values = (chi_enum / chi_denom).transpose()[:, np.newaxis]
    return chi_values, chi_values


def get_polarization(event, method = "noRT", cluster_method = None, between_topic=False):
    '''
    :param event: name of the event (str)
    :param method: "nofilter": use all tweets
                    "noRT" (default): ignore retweets only
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
    return get_values(event, data, token_partisanship_measure=chi_square, leaveout=True,
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
        topic_polarization[i] = tuple(get_values(event, data[data['topic'] == i], token_partisanship_measure=chi_square,
                                                 leaveout=False, default_score=0))

    cluster_method = method_name(cluster_method)
    with open(TWEET_DIR + event + '/' + event + '_chi_square_topic_polarization' + cluster_method + '.json', 'w') as f:
        f.write(json.dumps(topic_polarization))

if __name__ == "__main__":

    # for overall polarization

    event_polarization = {}
    method = sys.argv[1]
    cluster_method = None if len(sys.argv) < 3 else sys.argv[2]
    for e in events:
        event_polarization[e] = tuple(get_polarization(e, method, cluster_method))

    cluster_method = method_name(cluster_method)
    with open(OUTPUT_DIR + 'chi_square_' + method + cluster_method + '_leaveout.json', 'w') as f:
        f.write(json.dumps(event_polarization))


    # for between topic polarization
    '''
    between_topic_polarization = {}
    cluster_method = None if len(sys.argv) < 2 else sys.argv[1]
    for e in events:
        between_topic_polarization[e] = tuple(get_polarization(e, "clustered", cluster_method, between_topic=True))

    cluster_method = method_name(cluster_method)
    with open(OUTPUT_DIR + 'chi_square_between_topic_polarization' + cluster_method + '.json', 'w') as f:
        f.write(json.dumps(between_topic_polarization))
    '''

    # for within topic polarization
    '''
    topic_polarization = {}
    cluster_method = None if len(sys.argv) < 2 else sys.argv[1]
    for e in events:
        get_polarization_topics(e, cluster_method)
    '''