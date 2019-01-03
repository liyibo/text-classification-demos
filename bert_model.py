#!/usr/bin/python
# coding=utf8

"""
# Created : 2018/12/28
# Version : python2.7
# Author  : yibo.li 
# File    : bert_model.py
# Desc    : 
"""

import os
import tensorflow as tf
import numpy as np
from datetime import datetime

from bert import modeling
from bert import optimization
from bert.data_loader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

processors = {"cnews": CnewsProcessor}

tf.logging.set_verbosity(tf.logging.INFO)


class BertModel():
    def __init__(self, bert_config, num_labels, seq_length, init_checkpoint):
        self.bert_config = bert_config
        self.num_labels = num_labels
        self.seq_length = seq_length

        self.input_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [None, self.seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='segment_ids')
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
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
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")

            one_hot_labels = tf.one_hot(self.labels, depth=self.num_labels, dtype=tf.float32)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=one_hot_labels)
            self.loss = tf.reduce_mean(cross_entropy, name="loss")
            self.optim = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(one_hot_labels, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="acc")


def make_tf_record(output_dir, data_dir, vocab_file):
    tf.gfile.MakeDirs(output_dir)
    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    train_file = os.path.join(output_dir, "train.tf_record")
    eval_file = os.path.join(output_dir, "eval.tf_record")

    # save data to tf_record
    train_examples = processor.get_train_examples(data_dir)
    file_based_convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer, train_file)

    # eval data
    eval_examples = processor.get_dev_examples(data_dir)
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
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.

    if is_training:
        data = data.shuffle(buffer_size=50000)
        data = data.repeat(num_epochs)


    data = data.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size))
    return data


def get_test_example():
    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)

    # save data to tf_record
    examples = processor.get_test_examples(data_dir)

    features = get_test_features(examples, label_list, max_seq_length, tokenizer)

    return features



def evaluate(sess, model):
    """
    评估 val data 的准确率和损失
    """

    # dev data
    test_record = tf.data.TFRecordDataset("./model/bert/eval.tf_record")
    test_data = read_data(test_record, train_batch_size, False, 3)
    test_iterator = test_data.make_one_shot_iterator()
    test_batch = test_iterator.get_next()

    data_nums = 0
    total_loss = 0.0
    total_acc = 0.0
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
            loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
        except Exception as e:
            print(e)
            break

    return total_loss / data_nums, total_acc / data_nums


def main():
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    with tf.Graph().as_default():
        # train data
        train_record = tf.data.TFRecordDataset("./model/bert/train.tf_record")
        train_data = read_data(train_record, train_batch_size, True, 3)
        train_iterator = train_data.make_one_shot_iterator()

        model = BertModel(bert_config, num_labels, max_seq_length, init_checkpoint)
        sess = tf.Session()
        saver = tf.train.Saver()
        train_steps = 0
        val_loss = 0.0
        val_acc = 0.0
        best_acc_val = 0.0
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
                    _, train_loss, train_acc = sess.run([model.optim, model.loss, model.acc],
                                                        feed_dict=feed_dict)

                    if train_steps % 1000 == 0:
                        val_loss, val_acc = evaluate(sess, model)

                    if val_acc > best_acc_val:
                        # 保存最好结果
                        best_acc_val = val_acc
                        saver.save(sess, "./model/bert/model", global_step=train_steps)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    now_time = datetime.now()
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(train_steps, train_loss, train_acc, val_loss, val_acc, now_time, improved_str))
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
    labels = graph.get_operation_by_name('labels').outputs[0]
    is_training = graph.get_operation_by_name('is_training').outputs[0]
    loss = graph.get_operation_by_name('loss/loss').outputs[0]
    acc = graph.get_operation_by_name('accuracy/acc').outputs[0]

    data_len = len(features)
    batch_size = 12
    num_batch = int((len(features) - 1) / batch_size) + 1

    for i in range(num_batch):
        print(i)
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)

        batch_len = end_index-start_index

        _input_ids = np.array([data.input_ids for data in features[start_index:end_index]])
        _input_mask = np.array([data.input_mask for data in features[start_index:end_index]])
        _segment_ids = np.array([data.segment_ids for data in features[start_index:end_index]])
        _labels = np.array([data.label_id for data in features[start_index:end_index]])
        feed_dict = {input_ids: _input_ids,
                     input_mask: _input_mask,
                     segment_ids: _segment_ids,
                     labels: _labels,
                     is_training: False}
        test_loss, test_acc = sess.run([loss, acc], feed_dict=feed_dict)
        total_loss += test_loss * batch_len
        total_acc += test_acc * batch_len

    return total_loss / data_len, total_acc / data_len



def test():
    features = get_test_example()
    graph_path = "./model/bert/model-7000.meta"
    model_path = "./model/bert"
    graph = tf.Graph()
    saver = tf.train.import_meta_graph(graph_path, graph=graph)
    sess = tf.Session(graph=graph)
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    test_loss, test_acc = test_model(sess, graph, features)
    print("Test loss: %f, Test acc: %f" % (test_loss, test_acc))


if __name__ == "__main__":
    data_dir = "data/cnews"
    output_dir = "model/bert"
    task_name = "cnews"
    vocab_file = "./bert/chinese_model/vocab.txt"
    bert_config_file = "./bert/chinese_model/bert_config.json"
    init_checkpoint = "./bert/chinese_model/bert_model.ckpt"
    max_seq_length = 512
    learning_rate = 2e-5
    train_batch_size = 12
    num_train_epochs = 3
    num_labels = 10
    # make_tf_record(output_dir, data_dir, vocab_file)
    # main()
    test()

