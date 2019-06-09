import tensorflow as tf
import numpy as np
import trfl, gym

from collections import deque
from attentive_np.attentive_np import Attention, LatentModel
from attentive_np.cartpole_reader import *
from snail import supervised_snail

class QNetwork(object):

	def __init__(self, name, learning_rate=0.01, state_size=4,
				 action_size=2, hidden_size=128, batch_size=64):

		with tf.variable_scope(name):

			#Input placeholder
			self._target_x  = tf.placeholder(tf.float32, [None, state_size])

			# Action placeholder
			self._actions = tf.placeholder(tf.int32, [batch_size], name='actions')

			# Snail network. This is where all the work happens.
			self.output = supervised_snail(self._target_x, 1, hidden_size)
			self.output = tf.keras.layers.Dense(action_size, activation=None)(self.output)

			self.name = name

			self._targetQs = tf.placeholder(tf.float32, [batch_size, action_size], name='target')
			self.reward = tf.placeholder(tf.float32, [batch_size], name='reward')
			self.discount = tf.constant(0.99, shape=[batch_size], dtype=tf.float32, name='discount')

			q_loss, q_learning = trfl.qlearning(self.output, self._actions, self.reward,
													   self.discount, self._targetQs)
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
	def last_n(self, n):
		return [self.buffer[ii] for ii in range(len(self.buffer) - n,len(self.buffer))]

# FOR DOUBLE DQN ONLY
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
