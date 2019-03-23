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
# in this script, we look at the leave-out scores of users and the time of their first tweet

def user_wordcounts_and_timestamps(tweets, vocab):
    users = tweets.groupby('user_id')
    avg_timestamps = []
    first_timestamps = []
    dem_follows = []
    rep_follows = []
    row_idx = []
    col_idx = []
    data = []
    for group_idx, (u, group), in enumerate(users):
        avg_timestamps.append(np.mean(group['timestamp']))
        first_timestamps.append(group['timestamp'].iloc[0])
        dem_follows.append(group['dem_follows'].iloc[0])
        rep_follows.append(group['rep_follows'].iloc[0])
        word_indices = []
        for split in group['text']:
            count = 0
            prev = ''
            for w in split:
                if w == '':
                    continue
                if w in vocab:
                    word_indices.append(vocab[w])
                if count > 0:
                    bigram = prev + ' ' + w
                    if bigram in vocab:
                        word_indices.append(vocab[bigram])
                count += 1
                prev = w
        for k, v in Counter(word_indices).items():
            col_idx.append(group_idx)
            row_idx.append(k)
            data.append(v)
    return sp.csr_matrix((data, (col_idx, row_idx)), shape=(len(users), len(vocab))), np.array(first_timestamps),\
           np.array(avg_timestamps), np.array(dem_follows), np.array(rep_follows)

def user_leaveout_polarization(event):
    print(event)
    tweets = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['text', 'timestamp', 'user_id', 'dem_follows', 'rep_follows', 'remove', 'isRT'])
    tweets = filter_retweets(tweets)

    # clean data
    tweets['text'] = tweets['text'].astype(str).apply(clean_text, args=(False, event))

    # get vocab
    vocab = {w: i for i, w in enumerate(open(TWEET_DIR + event + '/' + event + '_vocab.txt', 'r').read().splitlines())}

    dem_tweets, rep_tweets = split_party(tweets)  # get partisan tweets

    # get word counts, timestamps and followee count for each user
    dem_counts, dem_first_timestamps, dem_avg_timestamps, dem_dem_follows, dem_rep_follows = user_wordcounts_and_timestamps(dem_tweets, vocab)
    rep_counts, rep_first_timestamps, rep_avg_timestamps, rep_dem_follows, rep_rep_follows = user_wordcounts_and_timestamps(rep_tweets, vocab)
    assert(dem_counts.shape[0] == len(dem_first_timestamps))

    dem_user_len = dem_counts.shape[0]
    del dem_tweets
    del rep_tweets
    del tweets
    gc.collect()

    all_counts = sp.vstack([dem_counts, rep_counts])

    wordcounts = all_counts.nonzero()[1]

    # filter words used by less than 1 person
    all_counts = all_counts[:, np.array([(np.count_nonzero(wordcounts == i) > 1) for i in range(all_counts.shape[1])])]

    dem_counts = all_counts[:dem_user_len, :]
    rep_counts = all_counts[dem_user_len:, :]
    del wordcounts
    del all_counts
    gc.collect()

    # filter users who did not use words from vocab
    dem_nonzero = set(dem_counts.nonzero()[0])
    rep_nonzero = set(rep_counts.nonzero()[0])
    dem_filter_idx = np.array([(i in dem_nonzero) for i in range(dem_counts.shape[0])])
    rep_filter_idx = np.array([(i in rep_nonzero) for i in range(rep_counts.shape[0])])
    dem_counts = dem_counts[dem_filter_idx, :]
    rep_counts = rep_counts[rep_filter_idx, :]
    dem_first_timestamps = list(dem_first_timestamps[dem_filter_idx])
    rep_first_timestamps = list(rep_first_timestamps[rep_filter_idx])
    dem_avg_timestamps = list(dem_avg_timestamps[dem_filter_idx])
    rep_avg_timestamps = list(rep_avg_timestamps[rep_filter_idx])
    dem_dem_follows = list(dem_dem_follows[dem_filter_idx])
    dem_rep_follows = list(dem_rep_follows[dem_filter_idx])
    rep_dem_follows = list(rep_dem_follows[rep_filter_idx])
    rep_rep_follows = list(rep_rep_follows[rep_filter_idx])

    # get leaveout scores
    dem_sum = dem_counts.sum(axis=0)
    rep_sum = rep_counts.sum(axis=0)
    rep_sum_total = rep_sum.sum()
    rep_q = rep_sum / rep_sum_total
    dem_sum_total = dem_sum.sum()
    dem_q = dem_sum / dem_sum_total
    dem_user_total = dem_counts.sum(axis=1)
    rep_user_total = rep_counts.sum(axis=1)

    dem_leaveouts = []
    for i, u in enumerate(dem_counts):  # for each user, exclude them and get the empirical phrase frequencies
        minus_user = dem_sum - u
        minus_user_sum = minus_user.sum()
        dem_q_minus_user = minus_user / minus_user_sum
        np.testing.assert_almost_equal(dem_q_minus_user.sum(), 1, decimal=5)
        user_weighted = u.T / dem_user_total[i]
        total_q = dem_q_minus_user + rep_q
        dem_leaveouts.append((1. - rep_q / total_q).dot(user_weighted)[0, 0])

    del dem_counts
    del dem_sum
    gc.collect()

    rep_leaveouts = []
    for i, u in enumerate(rep_counts):  # for each user, exclude them and get the empirical phrase frequencies
        minus_user = rep_sum - u
        minus_user_sum = minus_user.sum()
        rep_q_minus_user = minus_user / minus_user_sum
        np.testing.assert_almost_equal(rep_q_minus_user.sum(), 1, decimal=5)
        user_weighted = u.T / rep_user_total[i]
        total_q = rep_q_minus_user + dem_q
        rep_leaveouts.append((rep_q_minus_user / total_q).dot(user_weighted)[0,0])

    del rep_counts
    del rep_sum
    del rep_q
    del dem_q
    gc.collect()


    df = pd.DataFrame({'avg_timestamp': dem_avg_timestamps + rep_avg_timestamps,
                       'first_timestamp': dem_first_timestamps + rep_first_timestamps,
                       'leaveout_score': dem_leaveouts + rep_leaveouts,
                       'dem_follows': dem_dem_follows + rep_dem_follows,
                       'rep_follows': dem_rep_follows + rep_rep_follows,
                       'party': ['dem'] * len(dem_first_timestamps) + ['rep'] * len(rep_first_timestamps)})

    df['avg_timestamp'] = df['avg_timestamp'] - event_times[event]
    df['first_timestamp'] = df['first_timestamp'] - event_times[event]
    return df

if __name__ == "__main__":
    dfs = []
    for event in events:
        df = user_leaveout_polarization(event)
        df['event'] = event
        dfs.append(df)
    concat = pd.concat(dfs, ignore_index=True)
    concat.to_csv(TWEET_DIR + 'user_polarization.csv', index=False)



