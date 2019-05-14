import tensorflow as tf
import tensorflow_probability as tfp
import fundamentals

import numpy as np

import math

def fc_layer(inputs, units):
	fc = tf.keras.layers.Dense(units, activation = None,
		kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0))(inputs)
	normed = tf.keras.layers.BatchNormalization()(fc)
	activations = tf.nn.leaky_relu(normed)

	return activations

# One layer of a temporal convolution block.
def dense_block(inputs, d, filters):
	xf = tf.keras.layers.Conv1D(filters = filters, kernel_size = 2, dilation_rate = d, padding = 'causal',
		kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))(inputs)
	xf = tf.keras.layers.BatchNormalization()(xf)
	activations = tf.nn.leaky_relu(xf)
	
	activations = tf.keras.layers.Dropout(0.4)(activations)

	return tf.concat([inputs, activations], axis = -1)

# Temporal convolution block
def tc_block(inputs, seq_length, filters):
	log = math.log(seq_length) / math.log(2)
	z = inputs
	for i in range(1, math.ceil(log)):
		z = dense_block(z, 2 ** i, filters)

	return z

# Attention block
def attention_block(inputs, key_size, val_size):
	keys = tf.keras.layers.Dense(key_size, activation=None)(inputs)
	queries = tf.keras.layers.Dense(key_size, activation=None)(inputs)
	vals = tf.keras.layers.Dense(val_size, activation=None)(inputs)
	sqrt_k = math.sqrt(key_size)

	logits = tf.matmul(queries, tf.linalg.transpose(keys))
	logits = tf.keras.layers.Masking(mask_value=0)(logits)
	probs = tf.math.softmax(logits / sqrt_k)
	
	read = tf.matmul(probs, vals)

	read = tf.keras.layers.Dropout(0.4)(read)

	return tf.concat([inputs, read], axis = -1)

# Supervised Neural AttentIve metaLearner model
def supervised_snail(inputs, seq_length, dense_size):
	hidden = fc_layer(inputs, dense_size)
	
	hidden = tf.keras.layers.Dropout(0.4)(hidden)

	hidden = tc_block(hidden, seq_length, 32)
	hidden = tc_block(hidden, seq_length, 32)
	
	hidden = attention_block(hidden, 32, 32)
	
	hidden = tc_block(hidden, seq_length, 64)
	hidden = tc_block(hidden, seq_length, 64)
	
	hidden = tf.layers.Flatten()(hidden)
	
	hidden = fc_layer(hidden, dense_size)
	
	out = tf.keras.layers.Dropout(0.4)(hidden)

	return out

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
	train_x = train_x[::2,:,:]
	train_y = train_y[::2,:]
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

	logits = supervised_snail(X, t, dense_size, n_classes)
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

				#if step > 0 and (step % display_step == 0 or step == 1):
				#	print("Step " + str(step) + ", Minibatch loss: " + \
				#		"{:.4f}".format(l) + ", Training accuracy: " + \
				#		"{:.3f}%".format(a * 100))

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
