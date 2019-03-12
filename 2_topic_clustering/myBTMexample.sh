#!/usr/bin/env bash#!/bin/bash
# run an toy example for BTM

K=10   # number of topics

alpha=`echo "scale=3;50/$K"|bc`
beta=0.01
niter=1000
save_step=501

btm_dir=/Users/ddemszky/BTM/
input_dir=${btm_dir}myData/
output_dir=${btm_dir}output/
model_dir=${output_dir}model/
mkdir -p $output_dir/model

# the input docs for training
doc_pt_train=${input_dir}tweets_train.txt
doc_pt_all=${input_dir}tweets_all.txt

echo "=============== Index Docs ============="
# docs after indexing
dwid_pt_train=${output_dir}train_tweet_wids.txt
dwid_pt_all=${output_dir}all_tweet_wids.txt
# vocabulary file
vocab_pt=${output_dir}vocab.txt

python indexDocs.py $doc_pt_train $dwid_pt_train $vocab_pt
python indexDocs.py $doc_pt_all $dwid_pt_all $vocab_pt

## learning parameters p(z) and p(w|z)
echo "=============== Topic Learning ============="
W=`wc -l < $vocab_pt` # vocabulary size
make -C ../src
echo "../src/btm est $K $W $alpha $beta $niter $save_step $dwid_pt_train $model_dir"
../src/btm est $K $W $alpha $beta $niter $save_step $dwid_pt_train $model_dir

## infer p(z|d) for each doc
echo "================ Infer P(z|d)==============="
echo "../src/btm inf sum_b $K $dwid_pt_all $model_dir"
../src/btm inf sum_b $K $dwid_pt_all $model_dir

## output top words of each topic
echo "================ Topic Display ============="
python topicDisplay.py $model_dir $K $vocab_pt
