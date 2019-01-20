import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples //batch_size
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
#nn1
W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.add(tf.matmul(x,W1),b1))
L1_drop = tf.nn.dropout(L1,keep_prob)
#nn2
W2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.add(tf.matmul(L1_drop,W2),b2))
L2_drop = tf.nn.dropout(L2,keep_prob)
#nn3
W3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.add(tf.matmul(L2_drop,W3),b3))
L3_drop = tf.nn.dropout(L3,keep_prob)
#nn3
W4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.add(tf.matmul(L3_drop,W4),b4))

#均方误差
#loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

correct = tf.equal(tf.argmax(y,1),tf.arg_max(prediction,1))
accurancy = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(20):
        for batch in range(n_batch):
            batc_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batc_x,y:batch_y,keep_prob:1.0,keep_prob:0.7})
        #acc = sess.run(accurancy, feed_dict={x: batc_x, y: batch_y,keep_prob:1.0})
        #print(" epoch: ",epoch,"  ACC: ",acc,"  loss: ",sess.run(loss,feed_dict={x:batc_x,y:batch_y,keep_prob:1.0}))
        test_acc = sess.run(accurancy, feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob:1.0})
        train_acc = sess.run(accurancy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("Epoch ",epoch)
        print(" train_acc: ", train_acc)
        print(" test_acc: ", test_acc)


