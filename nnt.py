import tensorflow as tf
import numpy as np
from sklearn import datasets

# input data (100 phony data points)

# iris = datasets.load_iris()
# np.array(iris)
# print(iris.dtype)
# x_data = iris.data
# y_data = iris.target

x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

# constructing a linear model
b = tf.Variable(tf.zeros(1))
W = tf.Variable(tf.random_uniform([1, 2], -1, 1))
y = tf.matmul(W, x_data) + b

# gradient descen
loss = tf.reduce_mean(tf.square(y - y_data))
# learning rate
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# init
init = tf.global_variables_initializer()

# lauch the graph
sess = tf.Session()
sess.run(init)

# train -- fit the plane
for step in range(0, 200):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(W), sess.run(b))











