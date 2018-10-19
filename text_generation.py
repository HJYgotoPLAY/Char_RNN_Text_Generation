# -*- encoding:utf8 -*-
from model import CharRNN
import tensorflow as tf
import numpy as np

# 加载数据
with open('data/anna.txt', 'r') as f:
    text = f.read()
# 构建字符集合
vocab = set(text)
lstm_size = 512
vocab_size = len(vocab)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    model = CharRNN(vocab_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1,1))
            x[0,0] = vocab_to_int(c)
            feed = {model.inputs: x, model.keep_prob: 1., model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], feed_dict=feed)

    c = pick_top_n(preds, vocab_size)
    samples.append(int_to_vocab(c))

    for i in range(n_samples):
        x[0,0] = vocab_to_int(c)
        feed = {model.inputs: x, model.keep_prob: 1., model.initial_state: new_state}
        preds, new_state = sess.run([model.prediction, model.final_state], feed_dict=feed)
        c = pick_top_n(preds, vocab_size)
        samples.append(int_to_vocab(c))
    return ''.join(samples)


checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="The")
print(samp)

