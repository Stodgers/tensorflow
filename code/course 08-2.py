
#code from 04-1
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples //batch_size
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#nn
W = tf.Variable(tf.zeros([784,10]),dtype=tf.float32)
b = tf.Variable(tf.zeros([10]),dtype=tf.float32)
prediction = tf.nn.softmax(tf.add(tf.matmul(x,W),b))

#均方误差
loss = tf.reduce_mean(tf.square(y-prediction))
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

correct = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
accurancy = tf.reduce_mean(tf.cast(correct,tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    '''
    for epoch in range(11):
        for batch in range(n_batch):
            batc_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batc_x,y:batch_y})
        acc = sess.run(accurancy, feed_dict={x: batc_x, y: batch_y})
        print(" epoch: ",epoch,"  ACC: ",acc,"  loss: ",sess.run(loss,feed_dict={x:batc_x,y:batch_y}))
    saver.save(sess,'net/my_cpt.cpt')
    '''
    print( sess.run(accurancy, feed_dict={x: mnist.train.images, y: mnist.train.labels}))
    saver.restore(sess,'net/my_cpt.cpt')
    print( sess.run(accurancy, feed_dict={x: mnist.train.images, y: mnist.train.labels}))



