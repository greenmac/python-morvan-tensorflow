# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-01-classifier/
# https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf16_classification/full_code.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # 如果沒有下載數據包, 會幫你下載, 第二次會直接運行

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size])) # 定義一個矩陣, 可使用大寫開頭區分
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1) # biases為偏差值, tf.zeros创建的参数, 初始值不為0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases # tf.matmul矩陣乘法
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28*28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# add error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1])) # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer()) # 初始化模型的參數

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) # 不是學習整套的data, 這樣比較快速, 提取出來部分的資料, 本次是100個
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels)) # 計算準確度的功能