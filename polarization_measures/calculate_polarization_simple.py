#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from collections import Counter
import sys
import gc
import scipy.sparse as sp
import nltk
import random
import pandas as pd
import argparse
sys.path.append('..')
from helpers.funcs import *

sno = nltk.stem.SnowballStemmer('english')

RNG = random.Random()  # make everything reproducible
RNG.seed(config['SEED'])
events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()

parser = argparse.ArgumentParser(description='Computes polarization value between two groups of texts.')
parser.add_argument('-i','--input_file', help='Input file (should have the following columns: text, user_id, partisanship).')
parser.add_argument('-v','--vocab_file', help='Text file, where each line is a word in the vocabulary.')
parser.add_argument('-s','--stopword_file', help='Text file, where each line is a stopword to remove (it can be an empty file).')
parser.add_argument('-g1','--group1_name', help='Name of group 1 in the partisanship column.')
parser.add_argument('-g2','--group2_name', help='Name of group 2 in the partisanship column.')
parser.add_argument('-sw','--stem_words', help='Whether to stem words.', action="store_true")
args = vars(parser.parse_args())

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

def get_party_q(party_counts, exclude_user_id = None):
    user_sum = party_counts.sum(axis=0)
    if exclude_user_id:
        user_sum -= party_counts[exclude_user_id, :]
    total_sum = user_sum.sum()
    return user_sum / total_sum

def get_rho(dem_q, rep_q):
    return (rep_q / (dem_q + rep_q)).transpose()

def calculate_polarization(dem_counts, rep_counts):
    dem_user_total = dem_counts.sum(axis=1)
    rep_user_total = rep_counts.sum(axis=1)

    dem_user_distr = (sp.diags(1 / dem_user_total.A.ravel())).dot(dem_counts)  # get row-wise distributions
    rep_user_distr = (sp.diags(1 / rep_user_total.A.ravel())).dot(rep_counts)
    dem_no = dem_counts.shape[0]
    rep_no = rep_counts.shape[0]
    assert (set(dem_user_total.nonzero()[0]) == set(range(dem_no)))  # make sure there are no zero rows
    assert (set(rep_user_total.nonzero()[0]) == set(range(rep_no)))  # make sure there are no zero rows

    dem_q = get_party_q(dem_counts)
    rep_q = get_party_q(rep_counts)

    # apply measures via leave-out
    dem_addup = 0
    rep_addup = 0
    for i in range(dem_no):
        dem_leaveout_q = get_party_q(dem_counts, i)
        token_scores_dem = 1. - get_rho(dem_leaveout_q, rep_q)
        dem_addup += dem_user_distr[i, :].dot(token_scores_dem)[0, 0]
    for i in range(rep_no):
        rep_leaveout_q = get_party_q(rep_counts, i)
        token_scores_rep = get_rho(dem_q, rep_leaveout_q)
        rep_addup += rep_user_distr[i, :].dot(token_scores_rep)[0, 0]
    rep_val = 1 / rep_no * rep_addup
    dem_val = 1 / dem_no * dem_addup
    return 1/2 * (dem_val + rep_val)


def clean_text(text, stem=True):
    stop = set(open(args["stopword_file"], 'r').read().splitlines())
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
    words = [w for w in words if w not in stop]
    if stem:
        words = [sno.stem(w) for w in words]
    return words


def split_party(data):
    return data[data['partisanship'] == args["group1_name"]], data[data['partisanship'] == args["group2_name"]]


def get_values(data):
    """
    Measure polarization.
    :param data: dataframe with 'text' and 'user_id'
    :return:
    """
    # clean data
    data['text'] = data['text'].astype(str).apply(clean_text, args=(args["stem_words"]))

    dem_tweets, rep_tweets = split_party(data)  # get partisan tweets

    # get vocab
    vocab = {w: i for i, w in
             enumerate(open(args["vocab_file"], 'r').read().splitlines())}
    dem_counts = get_user_token_counts(dem_tweets, vocab)
    rep_counts = get_user_token_counts(rep_tweets, vocab)

    del dem_tweets
    del rep_tweets
    del data
    gc.collect()

    # make the prior neutral (i.e. make sure there are the same number of Rep and Dem users)
    dem_user_len = dem_counts.shape[0]
    rep_user_len = rep_counts.shape[0]
    if dem_user_len > rep_user_len:
        dem_subset = np.array(RNG.sample(range(dem_user_len), rep_user_len))
        dem_counts = dem_counts[dem_subset, :]
        dem_user_len = dem_counts.shape[0]
    elif rep_user_len > dem_user_len:
        rep_subset = np.array(RNG.sample(range(rep_user_len), dem_user_len))
        rep_counts = rep_counts[rep_subset, :]
        rep_user_len = rep_counts.shape[0]
    assert (dem_user_len == rep_user_len)

    all_counts = sp.vstack([dem_counts, rep_counts])

    wordcounts = all_counts.nonzero()[1]

    # filter words used by fewer than 2 people
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

    actual_val = calculate_polarization(dem_counts, rep_counts)
    all_counts = sp.vstack([dem_counts, rep_counts])
    del dem_counts
    del rep_counts
    gc.collect()

    index = np.arange(all_counts.shape[0])
    RNG.shuffle(index)
    all_counts = all_counts[index, :]

    random_val = calculate_polarization(all_counts[:dem_user_len, :], all_counts[dem_user_len:, :])
    print(actual_val, random_val, dem_user_len + rep_user_len)
    sys.stdout.flush()
    del all_counts
    gc.collect()

    print("Actual value: %.3f" % actual_val)
    print("Random value: %.3f" % random_val)
    print("Number of users: %d" % dem_user_len + rep_user_len)

def load_data(input_file):
    data = pd.read_csv(input_file)
    return data


if __name__ == "__main__":
    data = pd.read_csv(args["input_file"])
    get_values(data)