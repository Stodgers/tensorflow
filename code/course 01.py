import tensorflow as tf
import numpy as np
'''
input1 = tf.constant(1.0)
input2 = tf.constant(2.0)
input3 = tf.constant(3.0)

add = tf.add(input1,input2)
mul = tf.multiply(input1,input3)

with tf.Session() as sess:
    ans1 = sess.run(add)
    ans2 = sess.run(mul)
    print(ans1,ans2)

#创建占位符

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
add = tf.add(x1,x2)
mul = tf.multiply(x1,x3)
with tf.Session() as sess:
    ans1 = sess.run(add,feed_dict={x1:[1,2],x2:[2,3]})
    ans2 = sess.run(mul,feed_dict={x1:[1,2],x3:[2,3]})
    print(ans1,ans2)
'''
x_data = np.random.rand(10000)
y_data = x_data*0.1+0.2

b = tf.Variable(0.)
w = tf.Variable(0.)
y = w*x_data + b

loss = tf.reduce_mean(tf.pow(y-y_data,2))

optimizer = tf.train.AdagradOptimizer(learning_rate=0.2).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(200001):
        sess.run(optimizer)
        if epoch%10==0:
            print(epoch," : ",+sess.run(loss)," k :",sess.run(w)," b: ",sess.run(b))