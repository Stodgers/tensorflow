import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
xs = mnist.train.images[:2000]
xy = mnist.train.labels[:2000]

batch_size = 100
n_batch = mnist.train.num_examples//batch_size

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #strides=[a,b,c,d]
    #固定a=d=1，b代表x方向步长，y代表y方向上的步长
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return  tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#[-1,w,h,channels]
x_image = tf.reshape(x,[-1,28,28,1])

##Layer1
#5*5窗口 输入1个平面通道 32平面通道
w_conv1 = weight_variable([5,5,1,16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

##Layer2
w_conv2 = weight_variable([5,5,16,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

##fc1
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

h_fc2 = tf.matmul(h_fc1_drop,w_fc2)+b_fc2
prediction = tf.nn.softmax(h_fc2)
#loss&train
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
AC_ = tf.reduce_mean(tf.cast(correction,tf.float32))
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})

        ACs = sess.run(AC_,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("Epoch: ",epoch,"   acc: ",ACs)