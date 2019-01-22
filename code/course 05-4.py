#copy from 3-2  little fine_turning
import numpy
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
lr = tf.Variable(0.3, tf.float32)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

#路径
DIR = 'D:\pyy\\tensorflow\code'

#sess
sess = tf.Session()

image_num = 3000
#load picture
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]),trainable=False,name='embedding')


#name_scope
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784],'x_input')
    y = tf.placeholder(tf.float32, [None, 10],'y_input')

with tf.name_scope('input_shape'):
    image_shape_input = tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shape_input,10)

with tf.name_scope('layer1'):
    with tf.name_scope('weight1'):
        W1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1),name='weight111')
        variable_summaries(W1)
    with tf.name_scope('b1'):
        b1 = tf.Variable(tf.zeros([300]) + 0.1,name='b111')
        variable_summaries(b1)
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
        tf.summary.scalar('loss',loss)
    with tf.name_scope('train'):
        train = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    correct = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
    with tf.name_scope('acc'):
        accurancy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('acc', accurancy)

if tf.gfile.Exists(DIR+'/projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + '/projector/projector')
    tf.gfile.MkDir(DIR + '/projector/projector')
with open(DIR+'/projector/projector/metadata.tsv','w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:],1))
    for i in range(image_num):
        f.write(str(labels[i])+'\n')

#合并所有的summaries
merged = tf.summary.merge_all()
projector_writer = tf.summary.FileWriter(DIR + '/projector/projector',sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + '/projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + '/projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(projector_writer,config)

init = tf.global_variables_initializer()
sess.run(init)
writer = tf.summary.FileWriter('logs/',sess.graph)
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
for epoch in range(5):
     sess.run(tf.assign(lr,0.2*(0.99**epoch)))
     for batch in range(n_batch):
        batc_x,batch_y = mnist.train.next_batch(batch_size)
        summary,_ = sess.run([merged,train],feed_dict={x:batc_x,y:batch_y},options=run_options,run_metadata=run_metadata)

     writer.add_summary(summary,epoch)
     projector_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
     projector_writer.add_summary(summary, epoch)
     acc = sess.run(accurancy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
     print(" epoch: ",epoch,"  ACC: ",acc,"  loss: ",sess.run(loss,feed_dict={x: mnist.train.images, y: mnist.train.labels},options=run_options,run_metadata=run_metadata))

saver.save(sess, DIR + '/projector/projector/a_model.ckpt', global_step=5)
projector_writer.close
sess.close

