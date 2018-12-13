# https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow11_build_network.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size])) # 定義一個矩陣, 可使用大寫開頭區分
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1) # biases為偏差值, tf.zeros创建的参数, 初始值不為0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases # tf.matmul矩陣乘法
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# np.newaxis 在使用和功能上等价于 None，查看源码发现：newaxis = None，其实就是 None 的一个别名
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise # np.square(平方)

# tf.placeholder()就是代表占位符，这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

# 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])) # tf.reduce_sum求和

# GradientDescentOptimizer要小於1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer() # 初始化, 後面才能運算
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data) # scatter散點圖
plt.ion() # 本次运行请注释，全局运行不要注释
plt.show()
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        # to see the step improvement
        # print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        try:
            ax.lines.remove(lines[0]) # 去除第一條線, 要不然太多條了
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)        
        plt.pause(0.1)