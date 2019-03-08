#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import numpy as np
import random

config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']
TWEET_DIR = config['TWEET_DIR']
SAMPLE_SIZE = config['SAMPLE_SIZE_FOR_TOPICS']
RNG = random.Random()
RNG.seed(config['SEED'])
events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()

def get_samples(SAMPLE_SIZE):
    for event in events:
        # number of tweets that have embeddings
        N = len(np.load(TWEET_DIR + event + '/' + event + '_cleaned_and_partisan_indices.npy'))

        print(event, N)
        rand = RNG.sample(range(N), min(N, SAMPLE_SIZE))

        np.save(TWEET_DIR + event + '/' + event + '_indices_among_embeddings_for_getting_topics.npy', rand)
    return

if __name__ == "__main__":
    get_samples(SAMPLE_SIZE)