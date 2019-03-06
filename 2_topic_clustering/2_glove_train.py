#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mittens import GloVe
import pandas as pd
from scipy import sparse
import json

# embedding dimension
d = 50
config = json.load(open('../config.json', 'r'))
DATA_DIR = config['OUTPUT_DIR']

print('Loading file...')
coocc = sparse.load_npz(DATA_DIR + 'glove_cooccurrence.npz')
coocc = coocc.toarray()

with open(DATA_DIR + 'joint_vocab.txt', 'r') as f:
    vocab = f.read().splitlines()


print('Training model...')
glove_model = GloVe(d, max_iter=5000, learning_rate=0.1)
embeddings = glove_model.fit(coocc)

embeddings = pd.DataFrame(embeddings, index=vocab)

embeddings.to_csv(DATA_DIR + 'glove.'+str(d)+'d.csv', sep='\t')



