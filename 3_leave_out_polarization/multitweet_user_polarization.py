#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import json
import sys
import gc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import Counter
sys.path.append('..')
from helpers.funcs import *

config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
TWEET_DIR = config['TWEET_DIR']
events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()
event_times = json.load(open(INPUT_DIR + 'event_times.json', 'r'))

hour = 60 * 60
day = 24 * hour
split_by = 24 * hour  # split by day
no_splits = int((day / split_by) * 10)  # 10 days

def get_multitweet_users(event):
    data = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['user_id', 'text', 'dem_follows', 'rep_follows', 'timestamp', 'remove', 'isRT'])
    data = filter_retweets(data)
    data = data[data['dem_follows'] != data['rep_follows']]  # keep partisan users
    data['text'] = data['text'].astype(str).apply(clean_text, args=(False, event))
    data = data[data['text'].str.len() > 0]  # keep users who used words from vocab
    data['user_id'] = data['user_id'].astype(int)
    total_u = len(set(data['user_id']))
    buckets, _ = get_buckets(data, event_times[event], no_splits, split_by)
    user_sets = {}
    for i, b in enumerate(buckets):
        user_sets[i] = set(b['user_id'])
    concat = []
    for i, u in user_sets.items():
        concat.extend(list(u))
    multitweet_users = [str(u) for u, c in Counter(concat).items() if c > 1]
    multi_u = len(multitweet_users)
    print(multi_u, total_u, multi_u / total_u)
    contains = data['user_id'].isin(set(multitweet_users)).sum()
    print(contains, contains / len(data))
    with open(TWEET_DIR + event + '/' + event + '_multitweet_users.txt', 'w') as f:
        f.write('\n'.join(multitweet_users))
    return multi_u / total_u, contains / len(data)


if __name__ == "__main__":
    multi_us = []
    multi_prop = []
    for e in events:
        print(e)
        mu, mp = get_multitweet_users(e)
        multi_us.append(mu)
        multi_prop.append(mp)
    print(np.mean(multi_us), np.std(multi_us))
    print(np.mean(multi_prop), np.std(multi_prop))
