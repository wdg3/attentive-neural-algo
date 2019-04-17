import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections

NPRegressionDescription = collections.namedtuple(
	"NPRegressionDescription",
	("query", "target_y", "num_total_points", "num_context_points"))

class GPCurvesReader(object):

	def __init__(self,
				 batch_size,
				 max_num_context,
				 x_size=1,
				 y_size=1,
				 l1_scale=0.6,
				 sigma_scale=1.0,
				 random_kernel_parameters=True,
				 testing=False):

		self._batch_size = batch_size
		self._max_num_context = max_num_context
		self._x_size = x_size
		self._y_size = y_size
		self._l1_scale = l1_scale
		self._sigma_scale = sigma_scale
		self._random_kernel_parameters = random_kernel_parameters
		self._testing = testing

	def _gaussian_kernel(self, x_data, l1, sigma_f, sigma_noise=2e-2):

		num_total_points = tf.shape(x_data)[1]

		x_data1 = tf.expand_dims(x_data, axis=1)
		x_data2 = tf.expand_dims(x_data, axis=2)
		diff = x_data1 - x_data2

		norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])
		norm = tf.reduce_sum(norm, -1)

		kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)
		kernel += (sigma_noise**2) * tf.eye(num_total_points)

		return kernel

	def generate_curves(self):

		num_context = tf.random_uniform(
			shape=[], minval=3, maxval=self._max_num_context, dtype=tf.int32)

		if self._testing:
			num_target = 400
			num_total_points = num_target
			x_values = tf.tile(
				tf.expand_dims(tf.range(-2., 2., 1. / 100, dtype=tf.float32), axis=0),
				[self._batch_size, 1])
		else:
			num_target = tf.random_uniform(shape=(), minval=0,
										   maxval=self._max_num_context - num_context,
										   dtype=tf.int32)
			num_total_points = num_context + num_target
			x_values = tf.random_uniform(
				[self._batch_size, num_total_points, self._x_size], -2, 2)

		if self._random_kernel_parameters:
			l1 = tf.random_uniform([self._batch_size, self._y_size,
									self._x_size], 0.1, self._l1_scale)
			sigma_f = tf.random_uniform([self._batch_size, self._y_size],
										 0.1, self._sigma_scale)
		else:
			l1 = tf.ones(shape=[self._batch_size, self._y_size,
								self._x_size], 0.1, self._l1_scale)
			sigma_f = tf.ones(shape=[self._batch_size, self._y_size]) * self.sigma_scale

		kernel = self._gaussian_kernel(x_values, l1, sigma_f)

		cholesky = tf.cast(tf.cholesky(tf.cast(kernel, tf.float64)), tf.float32)

		y_values = tf.matmul(
			cholesky,
			tf.random.normal([self._batch_size, self._y_size, num_total_points, 1]))

		y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])

		if self._testing:
			target_x = x_values
			target_y = y_values

			idx = tf.random_shuffle(tf.range(num_target))
			context_x = tf.gather(x_values, idx[:num_context], axis=1)
			context_y = tf.gather(y_values, idx[:num_context], axis=1)

		else:
			target_x = x_values[:, :num_target + num_context, :]
			target_y = y_values[:, :num_target + num_context, :]

			context_x = x_values[:, :num_context, :]
			context_y = y_values[:, :num_context, :]

		query = ((context_x, context_y), target_x)

		return NPRegressionDescription(
			query=query,
			target_y=target_y,
			num_total_points=tf.shape(target_x)[1],
			num_context_points=num_context)