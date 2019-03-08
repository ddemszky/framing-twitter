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


def get_user_counts(tweets, vocab):
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

def get_leaveout_value(event, b):

    # clean data
    b['text'] = b['text'].astype(str).apply(clean_text, args=(False, event))

    # get vocab
    vocab = {w: i for i, w in enumerate(open(TWEET_DIR + event + '/' + event + '_vocab.txt', 'r').read().splitlines())}

    if len(b) < 100:   # fewer than a 100 tweets
        return 0.5, 0.5  # return these values when there is not enough data to make predictions on

    dem_tweets, rep_tweets = split_party(b)  # get partisan tweets
    dem_length = float(len(dem_tweets))
    rep_length = float(len(rep_tweets))

    dem_counts = get_user_counts(dem_tweets, vocab)
    rep_counts = get_user_counts(rep_tweets, vocab)

    dem_user_len = dem_counts.shape[0]
    if dem_user_len < 10 or rep_counts.shape[0] < 10:
        return 0.5, 0.5
    del dem_tweets
    del rep_tweets
    del b
    gc.collect()

    all_counts = sp.vstack([dem_counts, rep_counts])

    wordcounts = all_counts.nonzero()[1]

    # filter words used by less than 1 person
    all_counts = all_counts[:, np.array([(np.count_nonzero(wordcounts == i) > 1) for i in range(all_counts.shape[1])])]

    if all_counts.shape[1] < 50:   # fewer than 50 words in vocab
        return 0.5, 0.5
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
    dem_user_len = dem_counts.shape[0]

    if dem_counts.shape[0] < 10 or rep_counts.shape[0] < 10:   # fewer than 10 Reps or Dems
        return 0.5, 0.5

    pi_lo = leaveout(dem_counts, rep_counts)

    all_counts = sp.vstack([dem_counts, rep_counts])
    del dem_counts
    del rep_counts
    gc.collect()

    index = np.arange(all_counts.shape[0])
    RNG.shuffle(index)
    all_counts = all_counts[index, :]

    pi_lo_random = leaveout(all_counts[:dem_user_len, :], all_counts[dem_user_len:, :])
    print(pi_lo, pi_lo_random, dem_length + rep_length)
    sys.stdout.flush()
    del all_counts
    gc.collect()

    return pi_lo, pi_lo_random


