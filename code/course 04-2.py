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
W1 = tf.Variable(tf.truncated_normal([784,1000],stddev=0.1))
b1 = tf.Variable(tf.zeros([1000])+0.1)
L1 = tf.nn.relu(tf.add(tf.matmul(x,W1),b1))
L1_drop = tf.nn.dropout(L1,keep_prob)
#nn2
# W2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
# b2 = tf.Variable(tf.zeros([2000])+0.1)
# L2 = tf.nn.tanh(tf.add(tf.matmul(L1_drop,W2),b2))
# L2_drop = tf.nn.dropout(L2,keep_prob)
#nn3
# W3 = tf.Variable(tf.truncated_normal([1000,1000],stddev=0.1))
# b3 = tf.Variable(tf.zeros([1000])+0.1)
# L3 = tf.nn.tanh(tf.add(tf.matmul(L1_drop,W3),b3))
# L3_drop = tf.nn.dropout(L1,keep_prob)
#nn3
W4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = (tf.add(tf.matmul(L1_drop,W4),b4))

#均方误差
#loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

correct = tf.equal(tf.argmax(y,1),tf.arg_max(prediction,1))
accurancy = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(20):
        for batch in range(n_batch):
            batc_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batc_x,y:batch_y,keep_prob:1.0,keep_prob:1.0})
        #acc = sess.run(accurancy, feed_dict={x: batc_x, y: batch_y,keep_prob:1.0})
        #print(" epoch: ",epoch,"  ACC: ",acc,"  loss: ",sess.run(loss,feed_dict={x:batc_x,y:batch_y,keep_prob:1.0}))
        test_acc = sess.run(accurancy, feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob:1.0})
        train_acc = sess.run(accurancy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("Epoch ",epoch)
        print(" train_acc: ", train_acc)
        print(" test_acc: ", test_acc)

# Epoch  0
#  train_acc:  0.95878184
#  test_acc:  0.9547
# Epoch  1
#  train_acc:  0.9742909
#  test_acc:  0.9665
# Epoch  2
#  train_acc:  0.9826
#  test_acc:  0.9731
# Epoch  3
#  train_acc:  0.9862546
#  test_acc:  0.9756
# Epoch  4
#  train_acc:  0.98987275
#  test_acc:  0.9788
# Epoch  5
#  train_acc:  0.9922182
#  test_acc:  0.9778
# Epoch  6
#  train_acc:  0.9937091
#  test_acc:  0.9787
# Epoch  7
#  train_acc:  0.99552727
#  test_acc:  0.9796
# Epoch  8
#  train_acc:  0.9969818
#  test_acc:  0.9802
# Epoch  9
#  train_acc:  0.9977818
#  test_acc:  0.981
# Epoch  10
#  train_acc:  0.9983818
#  test_acc:  0.9814
# Epoch  11
#  train_acc:  0.9986727
#  test_acc:  0.9805
# Epoch  12
#  train_acc:  0.9990182
#  test_acc:  0.9808
# Epoch  13
#  train_acc:  0.99945456
#  test_acc:  0.982
# Epoch  14
#  train_acc:  0.99954545
#  test_acc:  0.9814
# Epoch  15
#  train_acc:  0.99972725
#  test_acc:  0.982
# Epoch  16
#  train_acc:  0.9998
#  test_acc:  0.9816
# Epoch  17
#  train_acc:  0.9997636
#  test_acc:  0.9821
# Epoch  18
#  train_acc:  0.99985456
#  test_acc:  0.9819
# Epoch  19
#  train_acc:  0.9999091
#  test_acc:  0.982
