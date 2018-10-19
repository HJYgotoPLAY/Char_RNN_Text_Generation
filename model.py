# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

# 构建输入层，输入层的size取决于batch_size(n_seqs*n_steps）
def bulid_inputs(num_seqs, num_steps):
    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')
    # 定义keep_prob参数用来控制dropout的保留结点数
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob

# 构建LSTM层
def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    """
    :param lstm_size: lstm cell 中隐层结点数
    :param num_layers: lstm层数
    :param batch_size: num_seqs*num_steps
    :param keep_prob: dropout保留结点数
    """
    stack_drop = []
    for i in range(num_layers):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        stack_drop.append(drop)
    cell = tf.nn.rnn_cell.MultiRNNCell(stack_drop)
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state

# 构建输出层
def build_output(lstm_output, in_size, out_size):
    seq_output = tf.concat(lstm_output, 1)
    x = tf.reshape(seq_output, [-1, in_size])
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        sotfmax_b = tf.Variable(tf.zeros(out_size))
    logits = tf.matmul(x, softmax_w) + sotfmax_b
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss

def bulid_optimizer(loss, learning_rate, grad_clip):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    return optimizer

class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                       lstm_size=128, num_layers=2, learning_rate=0.001,
                       grad_clip=5, sampling=False):
        # 如果sampling是True，则采用SGD
        if sampling == True:
            batch_size, num_steps = 1,1
        else:
            batch_size, num_steps = batch_size, num_steps
        tf.reset_default_graph()
        # 输入层
        self.inputs, self.targets, self.keep_prob = bulid_inputs(batch_size, num_steps)
        # LSTM层
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
        # 对输入进行one-hot编码
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        # 运行RNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        # 预测结果
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        # loss
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = bulid_optimizer(self.loss, learning_rate, grad_clip)
