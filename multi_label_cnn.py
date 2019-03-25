#!/usr/bin/python
# coding=utf8

"""
# Created : 2019/02/19
# Version : python2.7
# Author  : yibo.li 
# File    : multi_label_cnn.py
# Desc    : 
"""

import os
from datetime import datetime
import tensorflow as tf
from sklearn import metrics

from util.sent_process import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TextCNN():
    """
    文本分类，CNN模型
    """

    def __init__(self, seq_length, num_classes, vocab_size):
        """

        :param config:
        """
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.filter_sizes = [3, 4, 5]
        self.embedding_dim = 128
        self.num_filters = 128
        self.hidden_dim = 128

        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.drop_prob = tf.placeholder(tf.float32, name='drop_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='learn_rate')
        self.l2_loss = tf.constant(0.0)
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

        self.inference()

    def inference(self):
        """

        :return:
        """
        # 词向量映射
        with tf.name_scope("embedding"):
            embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-%s" % i):
                # conv layer
                conv = tf.layers.conv1d(embedding_inputs, self.num_filters, filter_size,
                                        padding='valid', activation=tf.nn.relu,
                                        kernel_regularizer=self.regularizer)
                # global max pooling
                pooled = tf.layers.max_pooling1d(conv, self.seq_length - filter_size + 1, 1)
                pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 2)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # # Add dropout
        # with tf.name_scope("dropout"):
        #     h_drop = tf.layers.dropout(h_pool_flat, self.drop_prob)

        with tf.name_scope("score"):
            fc = tf.layers.dense(h_pool_flat, self.hidden_dim, activation=tf.nn.relu, name='fc1')
            fc = tf.layers.dropout(fc, self.drop_prob)
            # classify
            self.logits = tf.layers.dense(fc, self.num_classes, name='fc2')
            self.logits = tf.identity(self.logits, name='logits')

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            #     logits=self.logits, labels=self.input_y)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            l2_loss = tf.losses.get_regularization_loss()
            self.loss = tf.reduce_mean(losses, name="loss")
            self.loss += l2_loss

            # optim
            self.optim = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)


def get_logits_label(logits):
    y_predict_labels = []
    for line in logits:
        line_label = [i for i in range(len(line)) if line[i] >= 0.50]
        # if len(line_label) < 1:
        #     line_label = [np.argmax(line)]
        y_predict_labels.append(line_label)
    return y_predict_labels


def get_target_label(eval_y):
    eval_y_short = []
    for line in eval_y:
        target = []
        for index, label in enumerate(line):
            if label > 0:
                target.append(index)
        eval_y_short.append(target)
    return eval_y_short


def compute_confuse_matrix(target_y, predict_y):
    """
    compute TP, FP, FN given target lable and predict label
    :param target_y:
    :param predict_y:
    :param label_dict {label:(TP,FP,FN)}
    :return: macro_f1(a scalar),micro_f1(a scalar)
    """
    # count number of TP,FP,FN for each class

    label_dict = {}
    for i in range(len(cat_to_id)):
        label_dict[i] = (0, 0, 0)

    for num in range(len(target_y)):
        targe_tmp = target_y[num]
        pre_tmp = predict_y[num]
        unique_labels = set(targe_tmp + pre_tmp)
        for label in unique_labels:
            TP, FP, FN = label_dict[label]
            if label in pre_tmp and label in targe_tmp:  # predict=1,truth=1 (TP)
                TP = TP + 1
            elif label in pre_tmp and label not in targe_tmp:  # predict=1,truth=0(FP)
                FP = FP + 1
            elif label not in pre_tmp and label in targe_tmp:  # predict=0,truth=1(FN)
                FN = FN + 1
            label_dict[label] = (TP, FP, FN)
    return label_dict


def evaluate(sess, model, x_, y_):
    """
    评估 val data 的准确率和损失
    """
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 64, shuffle=False)
    total_loss = 0.0
    y_pred = []
    y_target = []
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {model.input_x: x_batch, model.input_y: y_batch,
                     model.drop_prob: 0}
        loss, logits = sess.run([model.loss, model.logits], feed_dict=feed_dict)
        total_loss += loss * batch_len
        y_batch_pred = get_logits_label(logits)
        y_batch_target = get_target_label(y_batch)
        y_pred.extend(y_batch_pred)
        y_target.extend(y_batch_target)

    confuse_matrix = compute_confuse_matrix(y_target, y_pred)
    f1_micro, f1_macro = compute_micro_macro(confuse_matrix)
    print(f1_micro, f1_macro)
    f1_score = (f1_micro + f1_macro) / 2.0

    return total_loss / data_len, f1_score, confuse_matrix, y_pred, y_target


def compute_micro_macro(label_dict):
    f1_micro = compute_f1_micro(label_dict)
    f1_macro = compute_f1_macro(label_dict)
    return f1_micro, f1_macro


def compute_f1_micro(label_dict):
    """
    compute f1_micro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_micro: a scalar
    """
    TP_micro, FP_micron, FN_micro = compute_micro(label_dict)
    f1_micro = compute_f1(TP_micro, FP_micron, FN_micro)
    return f1_micro


def compute_f1(TP, FP, FN):
    """
    compute f1
    :param TP_micro: number.e.g. 200
    :param FP_micro: number.e.g. 200
    :param FN_micro: number.e.g. 200
    :return: f1_score: a scalar
    """
    precison = TP / (TP + FP + small_value)
    recall = TP / (TP + FN + small_value)
    f1_score = (2 * precison * recall) / (precison + recall + small_value)

    return f1_score


def compute_f1_macro(label_dict):
    """
    compute f1_macro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_macro
    """
    f1_dict = {}
    num_classes = len(label_dict)
    for label, tuplee in label_dict.items():
        TP, FP, FN = tuplee
        f1_score_onelabel = compute_f1(TP, FP, FN)
        f1_dict[label] = f1_score_onelabel
    f1_score_sum = 0.0
    for label, f1_score in f1_dict.items():
        f1_score_sum += f1_score
    f1_score = f1_score_sum / float(num_classes)
    return f1_score


def compute_micro(label_dict):
    """
    compute micro FP,FP,FN
    :param label_dict_accusation: a dict. {label:(TP, FP, FN)}
    :return:TP_micro,FP_micro,FN_micro
    """
    TP_micro, FP_micro, FN_micro = 0.0, 0.0, 0.0
    for label, tuplee in label_dict.items():
        TP, FP, FN = tuplee
        TP_micro = TP_micro + TP
        FP_micro = FP_micro + FP
        FN_micro = FN_micro + FN
    return TP_micro, FP_micro, FN_micro


def main():
    epochs = 50
    best_acc_f1 = 0.0  # 最佳验证集准确率
    train_steps = 0
    val_loss = 0.0
    val_f1 = 0.0
    with tf.Graph().as_default():
        cnn_model = TextCNN(seq_length, num_classes, vocab_size)
        saver = tf.train.Saver()
        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                print('Epoch:', epoch + 1)
                batch_train = batch_iter(X_train, y_train, 64)
                for x_batch, y_batch in batch_train:
                    train_steps += 1
                    learn_rate = 0.001
                    # if epoch > 5:
                    #     learn_rate = 0.0001
                    # learning rate vary
                    feed_dict = {cnn_model.input_x: x_batch, cnn_model.input_y: y_batch,
                                 cnn_model.drop_prob: 0.5, cnn_model.learning_rate: learn_rate}

                    _, train_loss = sess.run([cnn_model.optim, cnn_model.loss],
                                             feed_dict=feed_dict)

                    if train_steps % 100 == 0:
                        val_loss, val_f1, confuse_matrix, y_pred, y_target \
                            = evaluate(sess, cnn_model, X_test, y_test)

                    if val_f1 > best_acc_f1:
                        # 保存最好结果
                        best_acc_f1 = val_f1
                        last_improved = train_steps
                        saver.save(sess, "./model/ind_all_label/model", global_step=train_steps)
                        for i in range(len(cat_to_id)):
                            print(confuse_matrix[i])
                        # for i in range(len(y_pred)):
                        #     print(y_target[i], y_pred[i])
                        # improved_str = '*'
                    else:
                        improved_str = ''
                    now_time = datetime.now()
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Val Loss: {2:>6.2}, ' \
                          'Val F1: {3:>6.2}, Time: {4}'
                    print(msg.format(train_steps, train_loss, val_loss, val_f1, now_time))


def test_model(sess, graph, X):
    """

    :param sess:
    :param graph:
    :param x_:
    :param y_:
    :return:
    """
    batch_eval = test_batch_iter(X, 64)
    input_x = graph.get_operation_by_name('input_x').outputs[0]
    drop_prob = graph.get_operation_by_name('drop_prob').outputs[0]
    logits = graph.get_operation_by_name('score/logits').outputs[0]

    y_preds = []
    for x_batch in batch_eval:
        feed_dict = {input_x: x_batch, drop_prob: 0}
        y_logits = sess.run(logits, feed_dict=feed_dict)
        y_batch_pred = get_logits_label(y_logits)
        y_preds.extend(y_batch_pred)
    return y_preds


def test():
    X, txt_data = process_test_file(data_path, word_to_id, cat_to_id)
    graph_path = "./model/ind_all_label/model-7700.meta"
    model_path = "./model/ind_all_label"
    graph = tf.Graph()
    saver = tf.train.import_meta_graph(graph_path, graph=graph)
    sess = tf.Session(graph=graph)
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    # for op in graph.get_operations():
    #     print(op.name)
    y_pred = test_model(sess, graph, X)
    dst_file = open("test_data.txt", "a")
    for i in range(len(y_pred)):
        labels = []
        if y_pred[i]:
            labels = [id_to_cat[j] for j in y_pred[i]]
            print(labels)
        if labels:
            dst_file.write(",".join(labels) + "\t" + txt_data[i] + "\n")
        else:
            dst_file.write(" " + "\t" + txt_data[i] + "\n")
    dst_file.close()


if __name__ == "__main__":
    word_to_id, id_to_word = word_2_id("ind_voc.txt")
    cat_to_id, id_to_cat = cat_2_id("label_names.txt")
    data_path = "data.xlsx"
    X, y = process_file(data_path, word_to_id, cat_to_id)
    test_num = int(len(X) * 0.1)
    X_train, y_train = X[test_num:], y[test_num:]
    X_test, y_test = X[:test_num], y[:test_num]
    vocab_size = 3000
    seq_length = 256
    num_classes = len(cat_to_id)
    small_value = 0.00001

    # main()
    test()
