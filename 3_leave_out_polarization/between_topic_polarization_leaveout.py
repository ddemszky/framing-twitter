#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
from calculate_leaveout_polarization import *
sys.path.append('..')
from helpers.funcs import *

config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']
TWEET_DIR = config['TWEET_DIR']
NUM_CLUSTERS = config['NUM_CLUSTERS']
RNG = random.Random()  # make everything reproducible
RNG.seed(config['SEED'])
events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()

def user_topic_counts(tweets):
    # user-based
    users = tweets.groupby('user_id')
    row_idx = []
    col_idx = []
    data = []
    for group_idx, (u, group), in enumerate(users):
        for k, v in Counter(group['topic']).items():
            col_idx.append(group_idx)
            row_idx.append(k)
            data.append(v)
    return sp.csr_matrix((data, (col_idx, row_idx)), shape=(len(users), NUM_CLUSTERS))

def get_polarization(event, cluster_method = None):
    '''

    :param event: name of the event (str)
    :param cluster_method: None, "relative" (we use this in the paper) or "absolute" (see 5_assign_tweets_to_clusters.py); must have relevant files
    :return: tuple: (true value, random value)
    '''
    data = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['user_id', 'dem_follows', 'rep_follows'])
    data = get_cluster_assignments(event, data, cluster_method)

    print(event, len(data))

    return get_values(event, data, between_topic=True, between_topic_count_func=user_topic_counts)

if __name__ == "__main__":
    between_topic_polarization = {}
    cluster_method = None if len(sys.argv) < 2 else sys.argv[1]
    for e in events:
        between_topic_polarization[e] = tuple(get_polarization(e, cluster_method))

    cluster_method = method_name(cluster_method)
    with open(OUTPUT_DIR + 'between_topic_polarization' + cluster_method + '.json', 'w') as f:
        f.write(json.dumps(between_topic_polarization))



