# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:54:40 2017

@author: celes
"""
import re
import os
import jieba
import collections
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences


def split_title(df_title):
    title_list = []
    title_max_len = 0
    pattern = '[\u4e00-\u9fff]+'
    stop_words_list = ['的','了','和','是','就','都','而','及','與','著','一個','沒有','我們','你們','妳們','他們','她們','是否']
    for line in df_title:
        line_seg = jieba.cut(line, cut_all=False)
        tokens = [_ for _ in line_seg if re.search(pattern, _) and _ not in stop_words_list]
        if len(tokens) > title_max_len:
            title_max_len = len(tokens)
        title_list.append(tokens)
    return title_list

def build_vocb(title_list):
    wordcounts = collections.Counter()
    for title in title_list:
        for word in title:
            wordcounts[word] += 1
    words = [wordcount[0] for wordcount in wordcounts.most_common() if wordcount[1]>3]
    word2index = {w:i+1 for i,w in enumerate(words)}
    voc_size = len(word2index) + 1
    return voc_size,word2index

def filter_words(title_list,word2index):
    filter_title_list = []
    title_max_len = 0
    for title in title_list:
        title = [word for word in title if word in word2index]
        if len(title) > title_max_len:
            title_max_len = len(title)
        filter_title_list.append(title)
    return filter_title_list, title_max_len

def text2mat(title_list, title_max_len, word2index):
    word2vec = Word2Vec.load(os.path.join('model', 'wiki_w2v_100.bin'))
    X = np.zeros((len(title_list), title_max_len, word2vec.vector_size))
    for i in range(len(title_list)):
        title = title_list[i]
        title_len = len(title)
        for j in range(title_len):
            try:
                X[i,title_max_len-j-1,:] = word2vec.wv[title[title_len-j-1]]
            except KeyError:
                pass
    return X        
def text2seq(word2index,title_list,title_max_len):
    texts = []
    for title in title_list:
        texts.append([word2index[_] for _ in title])
    seqs = pad_sequences(texts,maxlen = title_max_len)
    return seqs




    