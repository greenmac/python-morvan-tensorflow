# https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow7_variable.py
# https://github.com/MorvanZhou/Tensorflow-Tutorial/tree/master/tutorial-contents

import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 如果定义 Variable, 就一定要 initialize
init = tf.global_variables_initializer() # must have if define variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))