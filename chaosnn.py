 #-*- coding:utf-8 -*-

import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import imput_data

start_time = time.time()
print start_time
print "Reading MNIST"

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
print "Done"


x = tf.placeholder(tf.float32,[None,784])

W=tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros[10])

y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder(tf.float32,[None,10])
corss_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "Start"

for i in xrange(1000):
  batch_xs,batch_ys = mnist.train.next_batch(100)
  sess.run(train_setep,feed_dict={x:batch_xs,y_:batch_ys})

print "Finish"

correct_predicition = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

print "Accuracy"
print sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})

end_time = time.time()
elapsed_time = end_time - start_time
print elapsed_time

 
