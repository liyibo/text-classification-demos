#!/usr/bin/python
# coding=utf8

"""
# Created : 2019/02/19
# Version : python2.7
# Author  : yibo.li 
# File    : sent_process.py
# Desc    : multi_label data process
"""
import re
import random
import numpy as np
import pandas as pd
from collections import defaultdict

sub_patt = re.compile("\-*\d[\d\,\.]*%*")
han_patt = re.compile("[\u3400-\u9fa5]")


def read_data(data_path):
    data = pd.read_excel(data_path)
    data = data.fillna("")
    data_label = []
    for line in data.values:
        if line[0]:
            data_label.append(line.tolist())
    return data_label


def read_test_data(data_path):
    data = pd.read_excel(data_path)
    data = data.fillna("")
    data_label = []
    for line in data.values:
        if not line[0]:
            data_label.append(line[1])
    return data_label


def word_2_id(vocab_dir):
    """

    :param vocab_dir:
    :return:
    """
    with open(vocab_dir) as f:
        words = [_.strip() for _ in f.readlines()]

    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def cat_2_id(vocab_dir):
    """

    :return:
    """
    with open(vocab_dir) as f:
        categories = [_.strip() for _ in f.readlines()]
    cat_to_id = dict(zip(categories, range(len(categories))))
    id_to_cat = dict((v, k) for k, v in cat_to_id.items())

    return cat_to_id, id_to_cat


def process_file(data_dir, word_to_id, cat_to_id, seq_length=256):
    """

    :param data_dir:
    :param word_to_id:
    :param cat_to_id:
    :param seq_length:
    :return:
    """
    data_label = read_data(data_dir)
    random.shuffle(data_label)

    data_id, label_id = [], []
    for line in data_label:
        labels = line[0].split(",")
        tmp = [cat_to_id[i] for i in labels]
        y_pad = [0] * len(cat_to_id)
        for i in tmp:
            y_pad[i] = 1
        label_id.append(y_pad)
        sent = sub_patt.sub("圞", line[1])
        tmp = []
        for word in sent:
            if word in word_to_id:
                tmp.append(word_to_id[word])
            elif word == "圞":
                tmp.append(word_to_id["<NUM>"])
            else:
                tmp.append(word_to_id["<UNK>"])
        # pad to the required length
        if len(tmp) > seq_length:
            tmp = tmp[:seq_length]
        else:
            padding = [0] * (seq_length - len(tmp))
            tmp += padding
        data_id.append(tmp)

    return np.array(data_id), np.array(label_id)


def process_test_file(data_dir, word_to_id, cat_to_id, seq_length=256):
    """

    :param data_dir:
    :param word_to_id:
    :param cat_to_id:
    :param seq_length:
    :return:
    """
    data_label = read_test_data(data_dir)
    data_id = []
    for line in data_label:
        sent = sub_patt.sub("圞", line)
        tmp = []
        for word in sent:
            if word in word_to_id:
                tmp.append(word_to_id[word])
            elif word == "圞":
                tmp.append(word_to_id["<NUM>"])
            else:
                tmp.append(word_to_id["<UNK>"])
        # pad to the required length
        if len(tmp) > seq_length:
            tmp = tmp[:seq_length]
        else:
            padding = [0] * (seq_length - len(tmp))
            tmp += padding
        data_id.append(tmp)

    return np.array(data_id), data_label


def batch_iter(x, y, batch_size=64, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[shuffle_indices]
        y_shuffle = y[shuffle_indices]
    else:
        x_shuffle = x
        y_shuffle = y
    for i in range(num_batch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)
        yield (x_shuffle[start_index:end_index], y_shuffle[start_index:end_index])


def test_batch_iter(x, batch_size=64):
    """
    Generates a batch iterator for a dataset.
    """
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    for i in range(num_batch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)
        yield (x[start_index:end_index])


if __name__ == "__main__":
    word_to_id, id_to_word = word_2_id("ind_voc.txt")
    cat_to_id, id_to_cat = cat_2_id("label_names.txt")
    data_path = "ind_data.xlsx"
    process_file(data_path, word_to_id, cat_to_id)
