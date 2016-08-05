__author__ = 'esthervandenberg'

"""
This is an extension of the simple MNIST Tensorflow Tutorial at
(https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html)
"""

import tensorflow as tf
import utils as ut

# helper func
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#  init
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
input_size = 784
hidden1_size = 500
hidden2_size = 300
output_size = 10
learn_rate = 0.01

# input to hidden
W = weight_variable([input_size, hidden1_size])
b = bias_variable([hidden1_size])
h1 = tf.nn.tanh(tf.matmul(x, W) + b)
"""
# hidden1 to hidden2
W2 = weight_variable([hidden1_size, hidden2_size])
b2 = bias_variable([hidden2_size])
h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
"""
# hidden2 to output
W3 = weight_variable([hidden1_size, output_size])
b3 = bias_variable([output_size])
y = tf.nn.softmax(tf.matmul(h1, W3) + b3)

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y+ 1e-9))

# train algo
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# saver
saver = tf.train.Saver()

# eval
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

class NeuralNetwork:
    def __init__(self, train_x=None, train_y=None, test_x=None, test_y=None, model_path=None):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.model_path = model_path

    def get_acc(self, noise_level, state):
        # print accuracy
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, self.model_path + noise_level + state + ".ckpt")
        acc = sess.run(accuracy, feed_dict={x: self.test_x, y_: self.test_y})
        print('Accuracy for {} noise {} EM was {}'.formate(noise_level, state, acc))
        return acc

    def restored_prob_y(self, noise_level, state='before'):
        # get predictions from pretrained model
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, self.model_path + noise_level + state + ".ckpt")
        prob_y = sess.run(y, feed_dict={x: self.train_x})
        return prob_y

    def train_NN(self, labels=None, noise_level=None, save_state=None, batch_size=100, epoch_size=10):
        # train NN and return accuracy and predictions
        if save_state == 'before':
            print('Training with naive parameters...')
        if save_state == 'after':
            print('Training with updated parameters...')

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        batches_x, batches_y = ut.get_batches(self.train_x, labels, batch_size=batch_size, epoch_size=epoch_size) #
        for i in range(len(batches_x)):
          batch_xs, batch_ys = batches_x[i], batches_y[i]
          sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        prob_y = sess.run(y, feed_dict={x: self.train_x})
        acc = sess.run(accuracy, feed_dict={x: self.test_x, y_: self.test_y})
        save_path = saver.save(sess, self.model_path + noise_level + save_state + ".ckpt")
        print("Model saved in file: %s" %save_path)
        print("Accuracy was: %s" %acc)

        return acc, prob_y

