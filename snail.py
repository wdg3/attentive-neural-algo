import tensorflow as tf
import tensorflow_probability as tfp
import fundamentals

import numpy as np

import math

# One Fully connected layer for SNAIL architecture
def fc_layer(inputs, units, layer, prefix):
	bn = tf.keras.layers.BatchNormalization(name=prefix+'bn_' + str(layer))(inputs)
	fc = tf.keras.layers.Dense(units, name = prefix+'dense_' + str(layer), 
		activation = None, kernel_initializer = 'he_normal',
		kernel_regularizer = tf.keras.regularizers.l2(1e-5))(bn)

	activations = tf.keras.layers.Activation('relu', name=prefix+'relu_' + str(layer))(fc)

	return activations

# One layer of a temporal convolution block. As described in Mishra et. al., 2017.
def dense_block(inputs, d, filters, layer):
	# We use the "gated activation" sigmoid * tanh.
	# Batch norm and dropout are used in each block.

	# Sigmoid path
	xf = tf.keras.layers.BatchNormalization(name = 'bn_sig_' + layer)(inputs)
	xf = tf.keras.layers.Conv1D(filters = filters, kernel_size = 2, dilation_rate = d, padding = 'causal',
		kernel_initializer='he_normal', name='conv1d_sig_' + layer,
		kernel_regularizer=tf.keras.regularizers.l2(1e-5))(xf)
	xf = tf.keras.layers.Activation('sigmoid', name='sigmoid' + layer)(xf)

	# Tanh path
	xg = tf.keras.layers.BatchNormalization(name = 'bn_tanh_' + layer)(inputs)
	xg = tf.keras.layers.Conv1D(filters = filters, kernel_size = 2, dilation_rate = d, padding = 'causal',
		kernel_initializer='he_normal', name='conv1d__tanh_' + layer,
		kernel_regularizer=tf.keras.regularizers.l2(1e-5))(xg)
	xg = tf.keras.layers.Activation('tanh', name='tanh' + layer)(xg)
	
	activations = tf.keras.layers.Multiply()([xf, xg])
	activations = tf.keras.layers.Dropout(0.4, name='dropout_' + layer)(activations)

	return tf.keras.layers.Concatenate(name='concat_' + layer)([activations, inputs])

# Temporal convolution block
def tc_block(inputs, seq_length, filters, layer, prefix):
	log = math.log(seq_length) / math.log(2)
	z = inputs
	# Keep increasing dilation rate until desired level
	for i in range(1, math.ceil(log)):
		z = dense_block(z, 2 ** i, filters, prefix+str(layer) + '_' + str(i))

	return z

# Attention block as described in Mishra et. al. 2017
def attention_block(inputs):
	key_size = val_size=16 #K, V
	keys = tf.keras.layers.Dense(key_size, activation=None)(inputs)
	queries = tf.keras.layers.Dense(key_size, activation=None)(inputs)
	vals = tf.keras.layers.Dense(val_size, activation=None)(inputs)
	sqrt_k = math.sqrt(key_size)

	logits = tf.matmul(queries, tf.linalg.transpose(keys))
	# Temporal masking so we don't "see" information we shouldn't
	logits = tf.keras.layers.Masking(mask_value=0)(logits)
	probs = tf.math.softmax(logits / sqrt_k)
	
	read = tf.matmul(probs, vals)

	return tf.concat([inputs, read], axis = -1)

# Supervised Neural AttentIve metaLearner model.
# As in Mishra et. al. 2017, we use two repetitions of 2
# temporal convolution blocks followed by an attention block.
def supervised_snail(inputs, seq_length, dense_size, prefix):

	hidden = tc_block(inputs, seq_length, 32, 1, prefix)
	hidden = tc_block(hidden, seq_length, 32, 2, prefix)
	
	# We use the Keras lambda wrapper to turn our attention block in a layer
	# compatible with the Keras framework.
	hidden = tf.keras.layers.Lambda(attention_block, name=prefix+'attn_1')(hidden)
	
	hidden = tc_block(hidden, seq_length, 32, 4, prefix)
	hidden = tc_block(hidden, seq_length, 32, 5, prefix)
	
	hidden = tf.keras.layers.Lambda(attention_block, name=prefix+'attn_2')(hidden)

	hidden = tf.layers.Flatten(name = prefix+'flatten_post')(hidden)

	# We output a 256 length encoding vector in our final model.
	out = fc_layer(hidden, dense_size, 7, prefix)

	return out

# ONLY USED FOR EARLY SUPERVISED LEARNING TESTING. NOT USED IN FINAL MODEL.
def main():
	tf.logging.set_verbosity(tf.logging.ERROR)
	tf.reset_default_graph()

	# Number of timesteps to use per sample
	timesteps = 8

	# Load data
	path = '../data/quandl/xy_' + str(timesteps) + '.h5'
	((train_x, train_y), (test_x, test_y)) = fundamentals.get_data(path)
	train_x = train_x.reshape(train_x.shape[0], timesteps, train_x.shape[2])
	test_x  = test_x.reshape(test_x.shape[0], timesteps, test_x.shape[2])

	# Using every other sample for memory reasons
	train_x = train_x[::4,:,:]
	train_y = train_y[::4,:]
	print(np.mean(train_y))

	# Parameter definitions
	m = train_x.shape[0]
	t = train_x.shape[1]
	n_x = train_x.shape[2]
	n_classes = train_y.shape[1]
	dense_size = 512
	learning_rate = 1e-4
	batch_size = 128
	total_batch = int(m / batch_size)
	epochs = 100
	display_step = 1000

	X = tf.placeholder(tf.float32, [None, t, n_x])
	Y = tf.placeholder(tf.float32, [None, n_classes])

	logits = supervised_snail(X, t, dense_size)
	y_hat = tf.keras.layers.Dense(n_classes, activation=tf.nn.sigmoid)(logits)
	preds = tf.math.round(y_hat)
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = y_hat), axis = 0)[0]
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	init = tf.initialize_all_variables()
	print("All parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.global_variables()]))
	print("Trainable parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)
		for e in range(1, epochs + 1):
			# Learning rate decay
			if e > 20:
				learning_rate = 1e-4
			if e > 50:
				learning_rate = 1e-5
			if e > 80:
				learning_rate = 1e-6
			step = 0
			losses = []
			accs = []
			while ((step + 1) * batch_size) < m:
				batch_x = train_x[(step * batch_size):((step + 1) * batch_size)]
				batch_y = train_y[(step * batch_size):((step + 1) * batch_size)]

				_, l, p = sess.run([train_step, loss, preds], feed_dict = {X: batch_x, Y: batch_y})
				a = np.mean(p == batch_y)
				accs.append(a)
				losses.append(l)

				step += 1

			p = sess.run(preds, feed_dict={X: test_x, Y: test_y})
			val_acc = np.mean(p == test_y)
			avg_loss = np.mean(losses)
			avg_acc = np.mean(accs)
			print()
			print('Epoch {} complete. Average loss: {:.4f}'.format(e, avg_loss))
			print('Average training accuracy: {:.3f}%, Test accuracy: {:.3f}%'.format(avg_acc * 100, val_acc * 100))
			save_path = saver.save(sess, "../models/model.ckpt")
			print("Model saved in path: %s" % save_path)
			print()

if __name__ == '__main__':
	main()