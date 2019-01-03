#!/usr/bin/python
#coding=utf8

"""
# Created : 2018/12/26
# Version : python2.7
# Author  : yibo.li 
# File    : han_model.py
# Desc    : 
"""

import os
from datetime import datetime
import tensorflow as tf

from util.cnews_loader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TextHan(object):
    def __init__(self, seq_length, num_classes, vocab_size):
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = 128
        self.num_sentences = 10
        self.hidden_dim = 128
        self.context_dim = 256
        self.rnn_type = "lstm"
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='learn_rate')

        self.inference()

    def inference(self):

        def _get_cell():
            if self.rnn_type == "vanilla":
                return tf.nn.rnn_cell.BasicRNNCell(self.context_dim)
            elif self.rnn_type == "lstm":
                return tf.nn.rnn_cell.BasicLSTMCell(self.context_dim)
            else:
                return tf.nn.rnn_cell.GRUCell(self.context_dim)

        def _Bidirectional_Encoder(inputs, name):
            with tf.variable_scope(name):
                fw_cell = _get_cell()
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
                bw_cell = _get_cell()
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)
                (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                 cell_bw=bw_cell,
                                                                                 inputs=inputs,
                                                                                 dtype=tf.float32)
            return output_fw, output_bw

        def _attention(inputs, name):
            with tf.variable_scope(name):
                # 使用一个全连接层编码 GRU 的输出，相当于一个隐藏层
                # [batch_size,sentence_length,hidden_size * 2]
                hidden_vec = tf.layers.dense(inputs, self.hidden_dim * 2,
                                             activation=tf.nn.tanh, name='w_hidden')

                # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
                # [hidden_size * 2]
                u_context = tf.Variable(tf.truncated_normal([self.hidden_dim * 2]), name='u_context')
                # [batch_size,sequence_length]
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(hidden_vec, u_context),
                                                    axis=2, keep_dims=True), dim=1)
                # before reduce_sum [batch_size, sequence_length, hidden_szie*2]，
                # after reduce_sum [batch_size, hidden_size*2]
                attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)

            return attention_output

        # 词向量映射
        with tf.name_scope("embedding"):
            input_x = tf.split(self.input_x, self.num_sentences, axis=1)
            # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
            input_x = tf.stack(input_x, axis=1)
            embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_dim])
            # [None,num_sentences,sentence_length,embed_size]
            embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)
            # [batch_size*num_sentences,sentence_length,embed_size]
            sentence_len = int(self.seq_length / self.num_sentences)
            embedding_inputs_reshaped = tf.reshape(embedding_inputs,
                                                   shape=[-1, sentence_len, self.embedding_dim])
        with tf.name_scope("word_vec"):
            (output_fw, output_bw) = _Bidirectional_Encoder(embedding_inputs_reshaped, "word_vec")
            # [batch_size*num_sentences,sentence_length,hidden_size * 2]
            word_hidden_state = tf.concat((output_fw, output_bw), 2)

        with tf.name_scope("word_attention"):
            """
           attention process:
           1.get logits for each word in the sentence.
           2.get possibility distribution for each word in the sentence.
           3.get weighted sum for the sentence as sentence representation.
           """
            # [batch_size*num_sentences, hidden_size * 2]
            sentence_vec = _attention(word_hidden_state, "word_attention")

        with tf.name_scope("sentence_vec"):
            # [batch_size,num_sentences,hidden_size*2]
            sentence_vec = tf.reshape(sentence_vec, shape=[-1, self.num_sentences,
                                                           self.context_dim * 2])
            output_fw, output_bw = _Bidirectional_Encoder(sentence_vec, "sentence_vec")
            # [batch_size*num_sentences,sentence_length,hidden_size * 2]
            sentence_hidden_state = tf.concat((output_fw, output_bw), 2)

        with tf.name_scope("sentence_attention"):
            # [batch_size, hidden_size * 2]
            doc_vec = _attention(sentence_hidden_state, "sentence_attention")

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(doc_vec, self.keep_prob)

        with tf.name_scope("score"):
            # 分类器
            self.logits = tf.layers.dense(h_drop, self.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")  # 预测类别

        with tf.name_scope("optimize"):
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
    loss = graph.get_operation_by_name('optimize/loss').outputs[0]
    acc = graph.get_operation_by_name('accuracy/acc').outputs[0]

    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {input_x: x_batch, input_y: y_batch,
                     keep_prob: 1}
        test_loss, test_acc = sess.run([loss, acc], feed_dict=feed_dict)
        total_loss += test_loss * batch_len
        total_acc += test_acc * batch_len

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
        seq_length = max_length
        num_classes = 10
        vocab_size = 5000
        model = TextHan(seq_length, num_classes, vocab_size)
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
                        saver.save(sess, "./model/han/model", global_step=train_steps)
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
    graph_path = "./model/han/model-7500.meta"
    model_path = "./model/han"
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
    max_length = 600

    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, vocab_size)

    main()
    # test()