import tensorflow as tf
import numpy as np
import trfl, gym

from collections import deque

class QNetwork(object):

	def __init__(self, name, learning_rate=0.01, state_size=4,
				 action_size=2, hidden_size=10, batch_size=20):

		with tf.variable_scope(name):
			self._inputs = tf.placeholder(tf.float32, [None, state_size],
						   name='inputs')

			self._actions = tf.placeholder(tf.int32, [batch_size], name='actions')

			self.fc1 = tf.contrib.layers.fully_connected(self._inputs, hidden_size)
			self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)
			self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size)
			self.fc4 = tf.contrib.layers.fully_connected(self.fc3, hidden_size)

			self.output = tf.contrib.layers.fully_connected(self.fc4, action_size,
															activation_fn=None)

			self.name = name

			self._targetQs = tf.placeholder(tf.float32, [batch_size, action_size], name='target')
			self.reward = tf.placeholder(tf.float32, [batch_size], name='reward')
			self.discount = tf.constant(0.99, shape=[batch_size], dtype=tf.float32, name='discount')

			print(self.output.shape)

			q_loss, q_learning = trfl.double_qlearning(self.output, self._actions, self.reward,
													   self.discount, self._targetQs, self.output)
			self.loss = tf.reduce_mean(q_loss)
			self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

class Memory(object):

	def __init__(self, max_size=1000):

		self.buffer = deque(maxlen=max_size)

	def add(self, experience):
			self.buffer.append(experience)

	def sample(self, batch_size):
			idx = np.random.choice(np.arange(len(self.buffer)),
								   size=batch_size,
								   replace=False)

			return [self.buffer[ii] for ii in idx]

def copy_model_parameters(sess, estimator1, estimator2):

	e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.name)]
	e1_params = sorted(e1_params, key=lambda v: v.name)
	e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.name)]
	e2_params = sorted(e2_params, key=lambda v: v.name)

	update_ops = []
	for e1_v, e2_v in zip(e1_params, e2_params):
		op = e2_v.assign(e1_v)
		update_ops.append(op)

	sess.run(update_ops)
