import tensorflow as tf

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
