# https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/202_placeholder.py
# https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow8_feeds.py

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

# placeholder是希望輸入的時候再給他一個新的值
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))