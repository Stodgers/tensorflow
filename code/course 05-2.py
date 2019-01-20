#copy from 3-2  little fine_turning
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
lr = tf.Variable(0.3, tf.float32)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size
#name_scope
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784],'x_input')
    y = tf.placeholder(tf.float32, [None, 10],'y_input')

with tf.name_scope('layer1'):
    with tf.name_scope('weight1'):
        W1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1),name='weight111')
    with tf.name_scope('b1'):
        b1 = tf.Variable(tf.zeros([300]) + 0.1,name='b111')
    with tf.name_scope('L1'):
        L1 = tf.add(tf.matmul(x, W1), b1)
    L1 = tf.tanh(L1)

with tf.name_scope('layer2'):
    with tf.name_scope('weight2'):
        W2 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
    with tf.name_scope('b2'):
        b2 = tf.Variable(tf.zeros([10]) + 0.1)
    prediction = (tf.add(tf.matmul(L1, W2), b2))

#均方误差
#loss = tf.reduce_mean(tf.square(y-prediction))

#cross entropy
with tf.name_scope('loss-train'):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    with tf.name_scope('train'):
        train = tf.train.GradientDescentOptimizer(lr).minimize(loss)

correct = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
accurancy = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(5):
        sess.run(tf.assign(lr,0.2*(0.99**epoch)))
        for batch in range(n_batch):
            batc_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batc_x,y:batch_y})
        acc = sess.run(accurancy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print(" epoch: ",epoch,"  ACC: ",acc,"  loss: ",sess.run(loss,feed_dict={x: mnist.train.images, y: mnist.train.labels}))



