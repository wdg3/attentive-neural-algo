import numpy as np
import tensorflow as tf

import collections

CartpoleDataDescription = collections.namedtuple(
	"CartpoleDataDescription",
	("query", "target_y", "num_total_points", "num_context_points"))

class CartpoleData(object):

	def __init__(self,
				 batch_size,
				 max_num_context,
				 random_num_context=False,
				 x_size=4,
				 y_size=2,
				 testing=False):

		self._batch_size = batch_size
		self._max_num_context = max_num_context
		self._random_num_context = random_num_context
		self._x_size = x_size
		self._y_size = y_size
		self._testing = testing

	def generate_context(self, memory):
		"""Data delivery function.

		Inputs:
			memory: Memory object to sample for context.
		Outputs:
			A tuple holding context_x and context_y.
		"""
		if self._random_num_context:
			num_context = tf.random.uniform(
				shape=[], minval=3, maxval=self._max_num_context, dtype=tf.int32)
		else:
			num_context = self._max_num_context

		# Sample contexts from memory B times.
		contexts = np.array([memory.sample(num_context)
			for m in range(self._batch_size)])

		# (B, num_context) of n_x arrays to (B, num_context, n_x)
		context_x = np.array(contexts[:, :, 0])
		new_x = np.zeros((self._batch_size, num_context, self._x_size))
		for i in range(self._batch_size):
			for j in range(num_context):
				new_x[i, j, :] = context_x[i, j]
		context_x = tf.convert_to_tensor(new_x)

		context_a = contexts[:, :, 1]
		context_y = tf.one_hot(context_a, depth=self._y_size, axis=-1, dtype=tf.int32)

		return (context_x, context_y)