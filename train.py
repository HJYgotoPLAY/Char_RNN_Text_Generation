# -*- encoding:utf8 -*-
import time
import numpy as np
import tensorflow as tf
from utils import get_batches
from model import CharRNN

# 加载数据
with open('data/anna.txt', 'r') as f:
    text = f.read()
# 构建字符集合
vocab = set(text)
# 字符_数字映射字典
vocab_to_int = {c: i for i, c in enumerate(vocab)}
# 数字_字符映射字典
int_to_vocab = dict(enumerate(vocab))
# 对文本进行转码
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

# 初始化参数
batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5
epochs = 20
# 没n轮进行一次变量保存
save_every_n = 200
model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps, lstm_size=lstm_size, num_layers=num_layers, learning_rate=learning_rate)
saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    for e in range(epochs):
        # train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.initial_state: new_state,
                    model.keep_prob: keep_prob}
            batch_loss, new_state, _ = sess.run([model.loss, model.final_state, model.optimizer], feed_dict=feed)
            end = time.time()
            # control the print line
            if counter % 100 == 0:
                print('轮数: {}/{}... '.format(e + 1, epochs),
                      '训练步数: {}... '.format(counter),
                      '训练误差: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end - start)))

            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))




