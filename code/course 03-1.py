import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#用numpy生成200随机点
x_data = np.linspace(-0.5,0.5,2000)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

#plt.scatter(x_data,y_data)
#plt.show()

#两个占位符，给运行时输入
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])


# x*w+b=y
#nn1
weight_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 =tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.add(tf.matmul(x,weight_L1),biases_L1)
L1 = tf.nn.tanh(Wx_plus_b_L1)

#nn2
weight_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 =tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.add(tf.matmul(L1,weight_L2),biases_L2)
L2 = tf.nn.tanh(Wx_plus_b_L2)

#损失
loss = tf.reduce_mean(tf.square(y-L2))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(20000):
        sess.run(train,feed_dict={x:x_data,y:y_data})
        if (epoch+1)%50==0:
            print("loss: ",sess.run(loss,feed_dict={x:x_data,y:y_data}))
    prediction = sess.run(L2,feed_dict={x:x_data,y:y_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction,'r',lw=5)
    plt.show()