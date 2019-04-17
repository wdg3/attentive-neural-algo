import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections

def batch_mlp(input, output_sizes, variable_scope):

	batch_size, _, filter_size = input.shape.as_list()
	output = tf.reshape(input, (-1, filter_size))
	output.set_shape((None, filter_size))

	with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
		for i, size in enumerate(output_sizes[:-1]):
			output = tf.nn.relu(
				tf.layers.dense(output, size, name="layer_{}".format(i)))

		output = tf.layers.dense(
			output, output_sizes[-1], name="layer_{}".format(i + 1))

	output = tf.reshape(output, (batch_size, -1, output_sizes[-1]))

	return output

class DeterministicEncoder(object):

	def __init__(self, output_sizes, attention):

		self._output_sizes = output_sizes
		self._attention = attention

	def __call__(self, context_x, context_y, target_x):

		encoder_input = tf.concat([context_x, context_y], axis=-1)

		hidden = batch_mlp(encoder_input, self._output_sizes, "deterministic_encoder")

		with tf.variable_scope("deterministic_encoder", reuse=tf.AUTO_REUSE):
			hidden = self._attention(context_x, target_x, hidden)

		return hidden

class LatentEncoder(object):

	def __init__(self, output_sizes, num_latents):

		self._output_sizes = output_sizes
		self._num_latents = num_latents

	def __call__(self, x, y):

		encoder_input = tf.concat([x, y], axis=-1)

		hidden = batch_mlp(encoder_input, self._output_sizes, "latent_encoder")

		with tf.variable_scope("latent_encoder", reuse=tf.AUTO_REUSE):
			hidden = tf.nn.relu(
				tf.layers.dense(hidden,
								(self._output_sizes[-1] + self._num_latents) / 2,
								name="penultimate_layer"))