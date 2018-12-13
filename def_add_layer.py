import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size])) # 定義一個矩陣, 可使用大寫開頭區分
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1) # biases為偏差值, tf.zeros创建的参数, 初始值不為0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases # tf.matmul矩陣乘法
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs