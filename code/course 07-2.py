import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# 输入图片是28*28
n_inputs = 28  # 输入一行，一行有28个数据
max_time = 28  # 一共28行
lstm_size = 100  # 隐层单元
n_classes = 10  # 10个分类
batch_size = 50  # 每批次50个样本
n_batch = mnist.train.num_examples // batch_size  # 计算一共有多少个批次

# 这里的none表示第一个维度可以是任意的长度
x = tf.placeholder(tf.float32, [None, 784])
# 正确的标签
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# 定义RNN网络
def RNN(X, weights, biases):
    # inputs=[batch_size, max_time, n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定义LSTM基本CELL
    # lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)
    lstm_cell = rnn.BasicLSTMCell(lstm_size)
    # final_state[0]是cell state
    # final_state[1]是hidden_state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results

prediction = RNN(x, weights, biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)
correction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correction,tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs,batch_xy = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_xy})
        Acs = sess.run(acc,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Epoch: ",epoch,"  Acc: ",Acs)