# https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf15_tensorboard/full_code.py
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = "layer%s" % n_layer
    with tf.name_scope('layer_name'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W') # 定義一個矩陣, 可使用大寫開頭區分
            tf.summary.histogram(layer_name + '/weights', Weights) # 總結, 觀看變量
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b') # biases為偏差值, tf.zeros创建的参数, 初始值不為0
            tf.summary.histogram(layer_name + '/biases', biases) # 總結, 觀看變量
        with tf.name_scope('Wx_plus_b'):        
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases) # tf.matmul矩陣乘法
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs) # 總結, 觀看變量
        return outputs

# Make up some real data
# 在指定的间隔内返回均匀间隔的数字。返回num均匀分布的样本，在[start, stop]。这个区间的端点可以任意的被排除在外。
# np.newaxis 在使用和功能上等价于 None，查看源码发现：newaxis = None，其实就是 None 的一个别名
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise # np.square(平方)

# define placeholder for inputs to network
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# the error between predictiuon and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])) # tf.reduce_sum求和
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# 给所有训练图合并
merged = tf.summary.merge_all()
# 这个方法中的第二个参数需要使用sess.graph ， 因此我们需要把这句话放在获取session的后面。 这里的graph是将前面定义的框架信息收集起来，然后放在logs/目录下面
writer = tf.summary.FileWriter('logs/', sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
    writer.add_summary(result, i)
# 最后在你的terminal（终端）中 ，使用以下命令
# tensorboard --logdir logs
# http://localhost:6006
