#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
import tensorflow as tf

from util.cnews_loader import *
# from util.cnews_seg_loader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TextRNN(object):
    """文本分类，RNN模型"""

    def __init__(self, seq_length, num_classes, vocab_size):
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = 64
        self.num_layers = 2
        self.rnn_name = 'gru'
        self.hidden_dim = 128
        self.learning_rate = 1e-3

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.inference()

    def inference(self):

        def lstm_cell(hidden_dim):  # lstm核
            return tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)

        def gru_cell(hidden_dim):  # gru核
            return tf.contrib.rnn.GRUCell(hidden_dim)

        def dropout(rnn_name, hidden_dim, keep_prob):
            if (rnn_name == 'lstm'):
                cell = lstm_cell(hidden_dim)
            else:
                cell = gru_cell(hidden_dim)
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

        # 词向量映射
        with tf.name_scope("embedding"):
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout(self.rnn_name, self.hidden_dim, self.keep_prob)
                     for _ in range(self.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy, name="loss")
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

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
        model = TextRNN(seq_length, num_classes, vocab_size)
        saver = tf.train.Saver()
        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                print('Epoch:', epoch + 1)
                batch_train = batch_iter(x_train, y_train, 64)
                for x_batch, y_batch in batch_train:
                    train_steps += 1
                    # if epoch > 5:
                    #     learn_rate = 0.0001
                    # learning rate vary
                    feed_dict = {model.input_x: x_batch, model.input_y: y_batch,
                                 model.keep_prob: 0.8}

                    _, train_loss, train_acc = sess.run([model.optim, model.loss, model.acc],
                                                        feed_dict=feed_dict)

                    if train_steps % 500 == 0:
                        val_loss, val_acc = evaluate(sess, model, x_val, y_val)

                    if val_acc > best_acc_val:
                        # 保存最好结果
                        best_acc_val = val_acc
                        last_improved = train_steps
                        saver.save(sess, "./model/rnn/model", global_step=train_steps)
                        # saver.save(sess=session, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    now_time = datetime.now()
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(train_steps, train_loss, train_acc, val_loss, val_acc, now_time, improved_str))


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
    # loss = graph.get_operation_by_name('loss/loss').outputs[0]
    acc = graph.get_operation_by_name('accuracy/acc').outputs[0]

    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {input_x: x_batch, input_y: y_batch, keep_prob: 1}
        test_acc = sess.run(acc, feed_dict=feed_dict)
        # total_loss += test_loss * batch_len
        total_acc += test_acc * batch_len

    return total_loss / data_len, total_acc / data_len


def test():
    word_to_id, id_to_word = word_2_id(vocab_dir)
    cat_to_id, id_to_cat = cat_2_id()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, max_length)
    graph_path = "./model/rnn/model-5500.meta"
    model_path = "./model/rnn"
    graph = tf.Graph()
    saver = tf.train.import_meta_graph(graph_path, graph=graph)
    sess = tf.Session(graph=graph)
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    test_loss, test_acc = test_model(sess, graph, x_test, y_test)
    print("Test loss: %f, Test acc: %f" % (test_loss, test_acc))


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

    # main() # 93.56
    test()
