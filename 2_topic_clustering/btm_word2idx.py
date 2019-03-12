#!/usr/bin/env python
# coding=utf-8
# translate word into id in documents, a modified version of xiaohuiyan's code (https://github.com/xiaohuiyan/BTM)
import sys
import json
config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']

def indexFile(pt, res_pt, word2idx):
    print('index file: ' + str(pt))
    with open(res_pt, 'w') as wf:
        for l in open(pt):
            words = l.strip().split()
            wids = [word2idx[w] for w in words if w in word2idx]
            wf.write(' '.join(map(str, wids)) + '\n')
        print('write file: ' + str(res_pt))

if __name__ == '__main__':
    doc_pt = sys.argv[1]   # docs to be indexed
    dwid_pt = sys.argv[2]   # output file for indexed docs
    word2idx = json.load(open(OUTPUT_DIR + 'joint_vocab_nostop.json', 'r'))
    indexFile(doc_pt, dwid_pt, word2idx)
    print('n(w)=' + str(len(word2idx)))
