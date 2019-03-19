#!/usr/bin/python
# coding=utf8

"""
# Created : 2018/12/27
# Version : python2.7
# Author  : yibo.li 
# File    : dpcnn_model.py
# Desc    : 
"""

import os
from datetime import datetime
import tensorflow as tf
from sklearn import metrics

from util.cnews_loader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
TODO: the use of batch_normalization in the model is error, need to fixed
"""

class DPCNN():
    """
    文本分类，DPCNN模型
    """

    def __init__(self, seq_length, num_classes, vocab_size):
        """

        :param config:
        """
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        self.embedding_dim = 128
        self.num_filters = 250
        self.kernel_size = 3

        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='learn_rate')

        self.inference()

    def inference(self):
        """

        :return:
        """
        # 词向量映射
        with tf.name_scope("embedding"):
            embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            embedding_inputs = tf.expand_dims(embedding_inputs, axis=-1)  # [None,seq,embedding,1]
            # region_embedding  # [batch,seq-3+1,1,250]
            region_embedding = tf.layers.conv2d(embedding_inputs, self.num_filters,
                                                [self.kernel_size, self.embedding_dim])

            pre_activation = tf.nn.relu(region_embedding, name='preactivation')

        with tf.name_scope("conv3_0"):
            conv3 = tf.layers.conv2d(pre_activation, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        with tf.name_scope("conv3_1"):
            conv3 = tf.layers.conv2d(conv3, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        # resdul
        conv3 = conv3 + region_embedding
        with tf.name_scope("pool_1"):
            pool = tf.pad(conv3, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
            pool = tf.nn.max_pool(pool, [1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

        with tf.name_scope("conv3_2"):
            conv3 = tf.layers.conv2d(pool, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        with tf.name_scope("conv3_3"):
            conv3 = tf.layers.conv2d(conv3, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        # resdul
        conv3 = conv3 + pool
        pool_size = int((self.seq_length - 3 + 1)/2)
        conv3 = tf.layers.max_pooling1d(tf.squeeze(conv3, [2]), pool_size, 1)
        conv3 = tf.squeeze(conv3, [1]) # [batch,250]
        conv3 = tf.nn.dropout(conv3, self.keep_prob)

        with tf.name_scope("score"):
            # classify
            self.logits = tf.layers.dense(conv3, self.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)

            # l2_loss = tf.losses.get_regularization_loss()
            self.loss = tf.reduce_mean(cross_entropy, name="loss")
            # self.loss += l2_loss

            # optim
            self.optim = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)
        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="acc")


def evaluate(sess, model, x_, y_):
    """
    评估 val data 的准确率和损失
    """
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 64)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {model.input_x: x_batch, model.input_y: y_batch,
                     model.keep_prob: 1}
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def test_model(sess, graph, x_, y_):
    """

    :param sess:
    :param graph:
    :param x_:
    :param y_:
    :return:
    """
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 64)
    total_loss = 0.0
    total_acc = 0.0

    input_x = graph.get_operation_by_name('input_x').outputs[0]
    input_y = graph.get_operation_by_name('input_y').outputs[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
    loss = graph.get_operation_by_name('loss/loss').outputs[0]
    acc = graph.get_operation_by_name('accuracy/acc').outputs[0]
    y_pred = graph.get_operation_by_name('score/pred').outputs[0]

    y_label_cls = []
    y_pred_cls = []
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {input_x: x_batch, input_y: y_batch,
                     keep_prob: 1}
        test_loss, test_acc, batch_pred = sess.run([loss, acc, y_pred], feed_dict=feed_dict)
        total_loss += test_loss * batch_len
        total_acc += test_acc * batch_len

        y_label = np.argmax(y_batch, 1)
        y_pred_cls.extend(batch_pred.tolist())
        y_label_cls.extend(y_label.tolist())

    return total_loss/data_len, total_acc/data_len, y_pred_cls, y_label_cls


def main():
    word_to_id, id_to_word = word_2_id(vocab_dir)
    cat_to_id, id_to_cat = cat_2_id()

    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, max_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, max_length)

    epochs = 10
    best_acc_val = 0.0  # 最佳验证集准确率
    train_steps = 0
    val_loss = 0.0
    val_acc = 0.0
    with tf.Graph().as_default():
        seq_length = 512
        num_classes = 10
        vocab_size = 5000
        model = DPCNN(seq_length, num_classes, vocab_size)
        saver = tf.train.Saver()
        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                print('Epoch:', epoch + 1)
                batch_train = batch_iter(x_train, y_train, 64)
                for x_batch, y_batch in batch_train:
                    train_steps += 1
                    learn_rate = 0.001
                    # learning rate vary
                    feed_dict = {model.input_x: x_batch, model.input_y: y_batch,
                                 model.keep_prob: 0.5, model.learning_rate: learn_rate}

                    _, train_loss, train_acc = sess.run([model.optim, model.loss,
                                                         model.acc], feed_dict=feed_dict)

                    if train_steps % 500 == 0:
                        val_loss, val_acc = evaluate(sess, model, x_val, y_val)

                    if val_acc > best_acc_val:
                        # 保存最好结果
                        best_acc_val = val_acc
                        last_improved = train_steps
                        saver.save(sess, "./model/dpcnn/model", global_step=train_steps)
                        # saver.save(sess=session, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    now_time = datetime.now()
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(train_steps, train_loss, train_acc, val_loss, val_acc, now_time, improved_str))


def test():
    word_to_id, id_to_word = word_2_id(vocab_dir)
    cat_to_id, id_to_cat = cat_2_id()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, max_length)
    graph_path = "./model/dpcnn/model-5000.meta"
    model_path = "./model/dpcnn"
    graph = tf.Graph()
    saver = tf.train.import_meta_graph(graph_path, graph=graph)
    sess = tf.Session(graph=graph)
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    test_loss, test_acc, y_pred_cls, y_label_cls = test_model(sess, graph, x_test, y_test)
    print("Test loss: %f, Test acc: %f" % (test_loss, test_acc))

    # 评估
    print("Precision, Recall and F1-Score...")
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    print(metrics.classification_report(y_label_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_label_cls, y_pred_cls)
    print(cm)


if __name__ == "__main__":
    base_dir = "./data/cnews"
    train_dir = os.path.join(base_dir, 'cnews.train.txt')
    test_dir = os.path.join(base_dir, 'cnews.test.txt')
    val_dir = os.path.join(base_dir, 'cnews.val.txt')
    vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

    vocab_size = 5000
    max_length = 512

    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, vocab_size)

    main()
    # test()
