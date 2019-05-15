#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import re
import string
from collections import Counter
import numpy as np
import sys
import gc
import json
import scipy.sparse as sp
import nltk
import random
sys.path.append('..')
from helpers.funcs import *

sno = nltk.stem.SnowballStemmer('english')

config = json.load(open('../config.json', 'r'))
TWEET_DIR = config['TWEET_DIR']
RNG = random.Random()  # make everything reproducible
RNG.seed(config['SEED'])


def get_user_token_counts(tweets, vocab):
    # user-based
    users = tweets.groupby('user_id')
    row_idx = []
    col_idx = []
    data = []
    for group_idx, (u, group), in enumerate(users):
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
    return sp.csr_matrix((data, (col_idx, row_idx)), shape=(len(users), len(vocab)))

def get_user_topic_counts(tweets):
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

def leaveout(dem_counts, rep_counts):

    dem_sum = dem_counts.sum(axis=0)
    rep_sum = rep_counts.sum(axis=0)
    rep_sum_total = rep_sum.sum()
    rep_q = rep_sum / rep_sum_total
    np.testing.assert_almost_equal(rep_q.sum(), 1, decimal=5)
    dem_sum_total = dem_sum.sum()
    dem_q = dem_sum / dem_sum_total
    np.testing.assert_almost_equal(dem_q.sum(), 1, decimal=5)

    dem_user_total = dem_counts.sum(axis=1)
    rep_user_total = rep_counts.sum(axis=1)

    dem_addup = 0
    for i, u in enumerate(dem_counts):  # for each user, exclude them and get the empirical phrase frequencies
        minus_user = dem_sum - u
        minus_user_sum = minus_user.sum()
        dem_q_minus_user = minus_user / minus_user_sum
        np.testing.assert_almost_equal(dem_q_minus_user.sum(), 1, decimal=5)
        user_weighted = u.T / dem_user_total[i]
        total_q = dem_q_minus_user + rep_q
        dem_addup += (1. - rep_q / total_q).dot(user_weighted)[0, 0]
    dem_val = 1 / 2 * 1 / dem_counts.shape[0] * dem_addup

    del dem_counts
    del dem_sum
    gc.collect()

    rep_addup = 0
    for i, u in enumerate(rep_counts):  # for each user, exclude them and get the empirical phrase frequencies
        minus_user = rep_sum - u
        minus_user_sum = minus_user.sum()
        rep_q_minus_user = minus_user / minus_user_sum
        np.testing.assert_almost_equal(rep_q_minus_user.sum(), 1, decimal=5)
        user_weighted = u.T / rep_user_total[i]
        total_q = rep_q_minus_user + dem_q
        rep_addup += (rep_q_minus_user / total_q).dot(user_weighted)[0,0]
    rep_val = 1 / 2 * 1 / rep_counts.shape[0] * rep_addup

    pi_lo = dem_val + rep_val

    del rep_counts
    del rep_sum
    del rep_q
    del dem_q
    gc.collect()

    return pi_lo

def calculate_polarization(dem_counts, rep_counts):


def get_values(event, data, token_partisanship_measure = probability, leaveout = False, between_topic = False,
               default_score = 0.5):
    """
    Measure polarization.
    :param event: name of the event
    :param data: dataframe with 'text' and 'user_id'
    :param token_partisanship_measure: a function calculating token partisanship based on user-token counts
    :param leaveout: whether to use leave-out estimation
    :param between_topic: whether the estimate is between topics or tokens
    :param default_score: default token partisanship score
    :return:
    """
    if not between_topic:
        # clean data
        data['text'] = data['text'].astype(str).apply(clean_text, args=(False, event))

    dem_tweets, rep_tweets = split_party(data)  # get partisan tweets

    if not between_topic:
        # get vocab
        vocab = {w: i for i, w in
                 enumerate(open(TWEET_DIR + event + '/' + event + '_vocab.txt', 'r').read().splitlines())}
        dem_counts = get_user_token_counts(dem_tweets, vocab)
        rep_counts = get_user_token_counts(rep_tweets, vocab)
    else:
        dem_counts = get_user_topic_counts(dem_tweets)
        rep_counts = get_user_topic_counts(rep_tweets)

    dem_user_len = dem_counts.shape[0]
    rep_user_len = rep_counts.shape[0]
    if dem_user_len < 10 or rep_user_len < 10:
        return default_score, default_score, dem_user_len + rep_user_len  # return these values when there is not enough data to make predictions on
    del dem_tweets
    del rep_tweets
    del data
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

    dem_nonzero = set(dem_counts.nonzero()[0])
    rep_nonzero = set(rep_counts.nonzero()[0])
    dem_counts = dem_counts[np.array([(i in dem_nonzero) for i in range(dem_counts.shape[0])]), :]  # filter users who did not use words from vocab
    rep_counts = rep_counts[np.array([(i in rep_nonzero) for i in range(rep_counts.shape[0])]), :]
    del dem_nonzero
    del rep_nonzero
    gc.collect()

    # make the prior neutral (i.e. make sure there are the same number of Rep and Dem users)
    dem_user_len = dem_counts.shape[0]
    rep_user_len = rep_counts.shape[0]
    if dem_user_len > rep_user_len:
        dem_subset = np.random.choice(np.arange(dem_user_len), rep_user_len, replace=False)
        dem_counts = dem_counts[dem_subset, :]
        dem_user_len = dem_counts.shape[0]
    elif rep_user_len > dem_user_len:
        rep_subset = np.random.choice(np.arange(rep_user_len), dem_user_len, replace=False)
        rep_counts = rep_counts[rep_subset, :]
        rep_user_len = rep_counts.shape[0]
    assert (dem_user_len == rep_user_len)

    actual_val = calculate_polarization(dem_counts, rep_counts)

    all_counts = sp.vstack([dem_counts, rep_counts])
    del dem_counts
    del rep_counts
    gc.collect()

    index = np.arange(all_counts.shape[0])
    RNG.shuffle(index)
    all_counts = all_counts[index, :]

    random_val = calculate_polarization(all_counts[:dem_user_len, :], all_counts[dem_user_len:, :])
    print(actual_val, random_val, dem_user_len * 2)
    sys.stdout.flush()
    del all_counts
    gc.collect()

    return actual_val, random_val, dem_user_len * 2