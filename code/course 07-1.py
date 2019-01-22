import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import  projector
import numpy as np
import matplotlib.pyplot as plt
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
xs = mnist.train.images[:2000]
xy = mnist.train.labels[:2000]

batch_size = 100
n_batch = mnist.train.num_examples//batch_size

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x,W):
    #strides=[a,b,c,d]
    #固定a=d=1，b代表x方向步长，y代表y方向上的步长
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return  tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x_input')
    y = tf.placeholder(tf.float32,[None,10],name='y_input')
    with tf.name_scope('image'):
        # [-1,w,h,channels]
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 10)


##Layer1
#5*5窗口 输入1个平面通道 32平面通道
with tf.name_scope('Conv1'):
    with tf.name_scope('W_conv1'):
        w_conv1 = weight_variable([5, 5, 1, 16],name='W_conv1')
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([16],name='b_conv1')
    with tf.name_scope('h_conv1'):
        h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    with tf.name_scope('Pool_conv1'):
        h_pool1 = max_pool_2x2(h_conv1)

##Layer2
with tf.name_scope('Conv2'):
    with tf.name_scope('W_conv2'):
        w_conv2 = weight_variable([5, 5, 16, 64],name='W_conv2')
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64],name='b_conv2')
    with tf.name_scope('h_conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    with tf.name_scope('Pool_conv1'):
        h_pool2 = max_pool_2x2(h_conv2)

##fc1
with tf.name_scope('fc1'):
    with tf.name_scope('w_fc1'):
        w_fc1 = weight_variable([7 * 7 * 64, 1024],name='w_fc1')
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024],name='b_fc1')
    with tf.name_scope('h_flat_fc1'):
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    with tf.name_scope('h_fc1'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('h_fc1_drop_out'):
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#fc2
with tf.name_scope('fc1'):
    with tf.name_scope('w_fc2'):
        w_fc2 = weight_variable([1024,10],name='w_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10],name='b_fc2')
    with tf.name_scope('h_fc2'):
        h_fc2 = tf.matmul(h_fc1_drop,w_fc2)+b_fc2
    with tf.name_scope('Prediction_fc2'):
        prediction = tf.nn.softmax(h_fc2)
#loss&train
with tf.name_scope('loss_train'):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
with tf.name_scope('correction_acc'):
    with tf.name_scope('correction'):
        correction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    with tf.name_scope('Acc'):
        AC_ = tf.reduce_mean(tf.cast(correction,tf.float32))
        tf.summary.scalar('Acc',AC_)
meraged = tf.summary.merge_all()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _ = sess.run([train_step],feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
            train_summary = sess.run(meraged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            train_writer.add_summary(train_summary, batch*epoch)

            batch_xs, batch_ys = mnist.test.next_batch(batch_size)
            test_summary = sess.run(meraged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            test_writer.add_summary(test_summary, batch*epoch)

        ACs = sess.run(AC_,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("Epoch: ",epoch,"   acc: ",ACs)