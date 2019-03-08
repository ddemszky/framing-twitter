from __future__ import division
import numpy as np
import json

config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']
TWEET_DIR = config['TWEET_DIR']
events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()

# for BTM, we use the script from here: https://github.com/xiaohuiyan/BTM

def get_samples():
    tweets = []
    for event in events:
        all_tweets = open(TWEET_DIR + event + '/' + event + '_cleaned_text.txt', 'r').read().splitlines()
        N = len(all_tweets)
        print(event, N)

        # get tweets with embeddings
        idx1 = np.load(TWEET_DIR + event + '/' + event + '_partisan_indices_among_cleaned_indices.npy')
        all_tweets = [all_tweets[i] for i in idx1]

        # get sample for determining topics
        idx2 = np.load(TWEET_DIR + event + '/' + event + '_indices_among_embeddings_for_getting_topics.npy')
        all_tweets = [all_tweets[i] for i in idx2]

        tweets.extend(all_tweets)

    with open('/Users/ddemszky/BTM/sample-data/doc_info.txt', 'w') as f:
        f.write('\n'.join(tweets))

get_samples()