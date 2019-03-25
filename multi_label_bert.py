#!/usr/bin/python
# coding=utf8

"""
# Created : 2019/02/19
# Version : python2.7
# Author  : yibo.li 
# File    : multi_label_bert.py
# Desc    : 
"""

import os
import random
import numpy as np
from sklearn import metrics
from datetime import datetime

from bert import modeling
from bert import optimization
from bert.data_loader import *
import util.sent_process

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

processors = {"cnews": CnewsProcessor, "ind": IndProcessor}

tf.logging.set_verbosity(tf.logging.INFO)


class BertModel():
    def __init__(self, bert_config, num_labels, seq_length, init_checkpoint):
        self.bert_config = bert_config
        self.num_labels = num_labels
        self.seq_length = seq_length

        self.input_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [None, self.seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='segment_ids')
        self.labels = tf.placeholder(tf.float32, [None, self.num_labels], name='labels')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learn_rate')

        self.model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        self.inference()

    def inference(self):

        output_layer = self.model.get_pooled_output()

        with tf.variable_scope("loss"):
            def apply_dropout_last_layer(output_layer):
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
                return output_layer

            def not_apply_dropout(output_layer):
                return output_layer

            output_layer = tf.cond(self.is_training, lambda: apply_dropout_last_layer(output_layer),
                                   lambda: not_apply_dropout(output_layer))
            self.logits = tf.layers.dense(output_layer, self.num_labels, name='fc')
            self.logits = tf.identity(self.logits, name='logits')

            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            self.loss = tf.reduce_mean(losses, name="loss")
            self.optim = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)


def make_tf_record(output_dir, data_train, data_test, vocab_file):
    tf.gfile.MakeDirs(output_dir)
    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)

    train_file = os.path.join(output_dir, "train.tf_record")
    eval_file = os.path.join(output_dir, "eval.tf_record")

    # save data to tf_record
    train_examples = processor.get_train_examples(data_train)
    file_based_convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer, train_file)

    # eval data
    eval_examples = processor.get_dev_examples(data_test)
    file_based_convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer, eval_file)

    del train_examples, eval_examples


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def read_data(data, batch_size, is_training, num_epochs):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([89], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.

    if is_training:
        data = data.shuffle(buffer_size=15000)
        data = data.repeat(num_epochs)

    data = data.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size))
    return data


def evaluate(sess, model):
    """
    评估 val data 的准确率和损失
    """

    # dev data
    test_record = tf.data.TFRecordDataset("./model/bert2/eval.tf_record")
    test_data = read_data(test_record, train_batch_size, False, 3)
    test_iterator = test_data.make_one_shot_iterator()
    test_batch = test_iterator.get_next()

    data_nums = 0
    total_loss = 0.0
    y_pred = []
    y_target = []
    while True:
        try:
            features = sess.run(test_batch)
            feed_dict = {model.input_ids: features["input_ids"],
                         model.input_mask: features["input_mask"],
                         model.segment_ids: features["segment_ids"],
                         model.labels: features["label_ids"],
                         model.is_training: False,
                         model.learning_rate: learning_rate}

            batch_len = len(features["input_ids"])
            data_nums += batch_len
            # print(data_nums)
            loss, logits = sess.run([model.loss, model.logits], feed_dict=feed_dict)
            total_loss += loss * batch_len
            y_batch_pred = get_logits_label(logits)
            y_batch_target = get_target_label(features["label_ids"])
            y_pred.extend(y_batch_pred)
            y_target.extend(y_batch_target)
        except Exception as e:
            break

    confuse_matrix = compute_confuse_matrix(y_target, y_pred)
    f1_micro, f1_macro = compute_micro_macro(confuse_matrix)
    print(f1_micro, f1_macro)
    f1_score = (f1_micro + f1_macro) / 2.0

    return total_loss / data_nums, f1_score, confuse_matrix, y_pred, y_target


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
    for i in range(89):
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
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    with tf.Graph().as_default():
        # train data
        train_record = tf.data.TFRecordDataset("./model/bert2/train.tf_record")
        train_data = read_data(train_record, train_batch_size, True, 20)
        train_iterator = train_data.make_one_shot_iterator()

        model = BertModel(bert_config, num_labels, max_seq_length, init_checkpoint)
        sess = tf.Session()
        saver = tf.train.Saver()
        train_steps = 0
        val_loss = 0.0
        val_f1 = 0.0
        best_acc_f1 = 0.0
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            train_batch = train_iterator.get_next()
            while True:
                try:
                    train_steps += 1
                    features = sess.run(train_batch)
                    feed_dict = {model.input_ids: features["input_ids"],
                                 model.input_mask: features["input_mask"],
                                 model.segment_ids: features["segment_ids"],
                                 model.labels: features["label_ids"],
                                 model.is_training: True,
                                 model.learning_rate: learning_rate}
                    _, train_loss = sess.run([model.optim, model.loss], feed_dict=feed_dict)

                    if train_steps % 200 == 0:
                        val_loss, val_f1, confuse_matrix, y_pred, y_target = evaluate(sess, model)

                    if val_f1 > best_acc_f1:
                        # 保存最好结果
                        best_acc_f1 = val_f1
                        saver.save(sess, "./model/bert2/model", global_step=train_steps)
                        improved_str = '*'
                        for i in range(89):
                            print(confuse_matrix[i])
                    else:
                        improved_str = ''

                    now_time = datetime.now()
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Val Loss: {2:>6.2}, ' \
                          'Val F1: {3:>6.2}, Time: {4}'
                    print(msg.format(train_steps, train_loss, val_loss, val_f1, now_time))
                except Exception as e:
                    print(e)
                    break


def test_model(sess, graph, features):
    """

    :param sess:
    :param graph:
    :param features:
    :return:
    """

    total_loss = 0.0
    total_acc = 0.0

    input_ids = graph.get_operation_by_name('input_ids').outputs[0]
    input_mask = graph.get_operation_by_name('input_mask').outputs[0]
    segment_ids = graph.get_operation_by_name('segment_ids').outputs[0]
    is_training = graph.get_operation_by_name('is_training').outputs[0]
    loss = graph.get_operation_by_name('loss/loss').outputs[0]
    logits = graph.get_operation_by_name('loss/logits').outputs[0]

    data_len = len(features)
    batch_size = 24
    num_batch = int((len(features) - 1) / batch_size) + 1
    y_preds = []
    for i in range(num_batch):
        print(i)
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)
        batch_len = end_index - start_index
        _input_ids = np.array([data.input_ids for data in features[start_index:end_index]])
        _input_mask = np.array([data.input_mask for data in features[start_index:end_index]])
        _segment_ids = np.array([data.segment_ids for data in features[start_index:end_index]])
        feed_dict = {input_ids: _input_ids,
                     input_mask: _input_mask,
                     segment_ids: _segment_ids,
                     is_training: False}
        y_logits = sess.run(logits, feed_dict=feed_dict)
        y_batch_pred = get_logits_label(y_logits)
        y_preds.extend(y_batch_pred)
    return y_preds


def get_test_example(test_data):
    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)

    # save data to tf_record
    examples = processor.get_test_examples(test_data)

    features = get_test_features(examples, label_list, max_seq_length, tokenizer)

    return features, label_list


def test():
    features, label_list = get_test_example(test_data)
    graph_path = "./model/bert2/model-7400.meta"
    model_path = "./model/bert2"
    graph = tf.Graph()
    saver = tf.train.import_meta_graph(graph_path, graph=graph)
    sess = tf.Session(graph=graph)
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    y_pred = test_model(sess, graph, features)
    dst_file = open("test_data2.txt", "a")
    for i in range(len(y_pred)):
        labels = []
        if y_pred[i]:
            labels = [label_list[j] for j in y_pred[i]]
            print(labels)
        if labels:
            dst_file.write(",".join(labels) + "\t" + test_data[i] + "\n")
        else:
            dst_file.write(" " + "\t" + test_data[i] + "\n")
    dst_file.close()


if __name__ == "__main__":
    output_dir = "model/bert2"
    task_name = "ind"
    vocab_file = "./bert/chinese_model/vocab.txt"
    bert_config_file = "./bert/chinese_model/bert_config.json"
    init_checkpoint = "./bert/chinese_model/bert_model.ckpt"
    max_seq_length = 256
    learning_rate = 2e-5
    train_batch_size = 24
    num_train_epochs = 20
    num_labels = 89
    small_value = 0.0001
    data_dir = "ind_data.xlsx"
    data_label = ind_sent_process.read_data(data_dir)
    random.shuffle(data_label)
    test_num = int(len(data_label) * 0.1)
    data_train = data_label[test_num:]
    data_test = data_label[:test_num]
    # make_tf_record(output_dir, data_train, data_test, vocab_file)
    main()
    # test_data = ind_sent_process.read_test_data(data_dir)
    # test()
