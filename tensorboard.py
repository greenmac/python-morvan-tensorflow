# https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf14_tensorboard/full_code.py
import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W') # 定義一個矩陣, 可使用大寫開頭區分
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b') # biases為偏差值, tf.zeros创建的参数, 初始值不為0
        with tf.name_scope('Wx_plus_b'):        
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases) # tf.matmul矩陣乘法
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

# define placeholder for inputs to network
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between predictiuon and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])) # tf.reduce_sum求和

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# 这个方法中的第二个参数需要使用sess.graph ， 因此我们需要把这句话放在获取session的后面。 这里的graph是将前面定义的框架信息收集起来，然后放在logs/目录下面
writer = tf.summary.FileWriter('logs/', sess.graph)


# 最后在你的terminal（终端）中 ，使用以下命令
# tensorboard --logdir logs
# http://localhost:6006
