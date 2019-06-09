import tensorflow as tf
import numpy as np
import pandas as pd

import datetime, sys

import data_utils
from snail import supervised_snail, attention_block, tc_block
from keras import backend as K
import sp_tickers

import h5py

# Get a subset of the data correspending to a list of tickers
def get_subset(xt, xf, y, dates, ticker_subset):
	tickers = dates[:,0]
	mask = np.zeros(len(tickers), dtype=bool)
	# Build mask out ticker by ticker
	for t in ticker_subset:
		mask = mask | (tickers == t)
	return xt[mask], xf[mask], y[mask], dates[mask]

# Binary crossentropy weighted by magnitude of price movement
def weighted_crossentropy(y_true, y_pred):
	binary = tf.cast(y_true > 0, tf.int32) # Convert to binary labels
	weights = tf.abs(y_true)

	return tf.losses.sparse_softmax_cross_entropy(binary, y_pred, weights=weights)

# Unweighted version (not used in final model)
def unweighted_crossentropy(y_true, y_pred):
	binary = tf.cast(y_true > 0, tf.int32)

	return tf.losses.sparse_softmax_cross_entropy(binary, y_pred)

# Simple accuracy metric based on binary version of labels
def acc(y_true, y_pred):
	binary = tf.cast(y_true > 0, tf.int32)
	return tf.keras.metrics.sparse_categorical_accuracy(binary, y_pred)

# Mean prediction on batch (used for debugging, not used in final model)
def mean_pred(y_true, y_pred):
	return K.mean(y_pred, axis=0)[1]

# This is our maine framework for creating, training, and testing a tensorflow model
# using the Keras libaries.
class Model(object):
	# Initialize model with information about the shape of our data.
	def __init__(self, input_shape_A, input_shape_B, t_steps, f_steps, n_y):
		self.model = self.create_model(input_shape_A, input_shape_B, t_steps, f_steps, n_y)
		self.opt = tf.keras.optimizers.Adam() # Adam optimizer with default beta parameters
		# Compile the model with custom loss function and accuracy metric.
		self.model.compile(optimizer=self.opt,
			loss=weighted_crossentropy,
			metrics=[acc])
		self.drop_rate = 0.4 #Dropout 40% of connections during training

	# Dense layer consist of Batch Norm -> Dense -> ReLU -> Dropout,
	# with L2 regularization and He normal initialization
	def dense_layer(self, inputs, units, drop_rate, suffix):
		bn = tf.keras.layers.BatchNormalization(name='bn_' + suffix)(inputs)
		
		dense = tf.keras.layers.Dense(units, activation=None,
			kernel_regularizer=tf.keras.regularizers.l2(1e-5),
			kernel_initializer='he_normal',
			name='dense_' + suffix)(bn)
		
		relu = tf.keras.layers.Activation('relu', name='relu_' + suffix)(dense)
		outputs = tf.keras.layers.Dropout(drop_rate, name='dropout_' + suffix)(relu)

		return outputs

	# Our experimental model uses to parallel snail modules concatenated 
	def create_model(self, input_shape_A, input_shape_B, t_steps, f_steps, n_y):
		inputs_A  = tf.keras.layers.Input(name='input_A', shape=input_shape_A)
		snail_A = supervised_snail(inputs_A, t_steps, 256, 'tech_')

		inputs_B = tf.keras.layers.Input(name='input_B', shape=input_shape_B)
		snail_B = supervised_snail(inputs_B, f_steps, 256, prefix='fund_')

		merge = tf.keras.layers.Concatenate(name='combiner')([snail_A, snail_B])

		merge = self.dense_layer(merge, 128, 0.4, 'a')

		outputs = tf.keras.layers.Dense(2, activation='softmax', name='out')(merge)
		return tf.keras.models.Model(name='main', inputs=[inputs_A, inputs_B], outputs=outputs)

def backtest(xt, xf, y, dates, model, subset=False):
	tickers = dates[:,0]
	dates   = dates[:,1]

	# Sort all our inputs chronologically
	xt = xt[np.argsort(dates)]
	xf = xf[np.argsort(dates)]
	y = y[np.argsort(dates)]
	tickers = tickers[np.argsort(dates)]
	dates = dates[np.argsort(dates)]

	# Grab asset closes and Dow closes
	closes  = xt[:,-1,3]
	dji_closes = xt[:,-1,14]

	# Get lists of unique dates and tickers in our data
	combined = np.array([closes, dji_closes, tickers, dates, y]).T
	unique_dates = np.unique(dates)
	unique_tickers = np.unique(tickers)

	model_return = np.zeros(len(unique_dates))
	dji_return = np.zeros(len(unique_dates))
	dji_last = dji_closes[0]
	cum_model = np.ones(len(unique_dates))
	cum_dji = np.ones(len(unique_dates))

	# Step through each date in our set and find the mean return the model
	# gets on each asset for that date
	for i in range(len(unique_dates)):
		date = unique_dates[i]
		x_date = xt[dates==date]
		x_f_date = xf[dates==date]
		y_date = y[dates==date]
		dji_close = dji_closes[dates==date]
		dji_close = dji_close[0]
		y_date = np.array([-1*y_date, y_date]).T
		preds = model.predict([x_date, x_f_date])

		model_return[i] = np.mean(preds * y_date)
		dji_return[i] =(dji_close / dji_last) - 1
		dji_last = dji_close
		if i > 0:
			cum_model[i] = cum_model[i-1] * (1+(model_return[i] / 8))
			cum_dji[i] = cum_dji[i-1] * (1+dji_return[i])

	# Find the cumulative return
	capital = 1.0
	model_return = model_return / 8
	for ret in model_return:
		capital *= 1 + ret
	capital -= 1
	dji = 1.0
	for ret in dji_return:
		dji *= 1 + ret
	dji -= 1

	# Print info about return
	if subset == False:
		print('Test period: {} - {}\nDow Jones Industrial Average return: {:.3f}%\nModel return on all: {:.3f}%'.format(
			unique_dates[0], unique_dates[-1], dji * 100, capital * 100))
	else:
		print('Model return on selected stocks: {:.3f}%'.format(capital * 100))
	print('({} prediction examples)'.format(len(y)))

	if len(unique_tickers) > 1:
		return unique_dates, cum_model, cum_dji

	else:
		return unique_dates, cum_model, (closes / closes[0])

# Learning rate schedule to be passed as a callback during training
def lr_schedule(e):
	if e < 20:
		return 1e-3
	elif e < 40:
		return 1e-4
	else:
		return 1e-5

# Global parameters
EPOCHS     = 50
BATCH_SIZE = 128
LOG_DIR    = './experimental/'
T_STEPS=20
F_STEPS=8
Y_HORIZON = 32
EVERY_N = 1
DEMO = True
load = False

ticker_subset = sp_tickers.ticker_subset

tf.reset_default_graph()

# Load data
print('Loading data...', end='')
sys.stdout.flush()
# We unforunately have to use dummy data for the demo due to the private nature
# of our dataset
if DEMO:
	m_train = 10000
	m_test  = 1000
	fs = 8
	ts = 20
	fn = 210
	tn = 23

	x_train = np.random.randn(m_train, ts, tn)
	x_f_train = np.random.randn(m_train, fs, fn)
	y_train = np.mean(np.mean(x_train, axis=-1), axis=-1).reshape(m_train, 1)
	t_train = np.random.randn(m_train, 4)

	x_test = np.random.randn(m_test, ts, tn)
	x_f_test = np.random.randn(m_test, fs, fn)
	y_test = np.mean(np.mean(x_test, axis=-1), axis=-1).reshape(m_test, 1)
	t_test = np.random.randn(m_test, 4)

	y_next = y_test[:,0]
else:
	path = '../data/'
	((x_train, y_train),
		(x_test, y_test),
		(x_f_train, x_f_test),
		(t_train, t_test),
		y_next) = data_utils.load_dataset(path, Y_HORIZON)

	# Using every N samples for memory reasons
	if EVERY_N > 1:
		x_train = x_train[::EVERY_N,:,:]
		y_train = y_train[::EVERY_N]
		t_train = t_train[::EVERY_N,:]
		x_test = x_test[::EVERY_N,:,:]
		y_test = y_test[::EVERY_N]
		t_test = t_test[::EVERY_N,:]
		x_f_train = x_f_train[::EVERY_N,:,:]
		x_f_test = x_f_test[::EVERY_N,:,:]
		y_next = y_next[::EVERY_N]

	y_train = y_train.reshape(len(y_train), 1)
	y_test = y_test.reshape(len(y_test), 1)

print('done.')

# Keras callback for backtesting model on test data after each training epoch.
# Compares model return to actual returns.
class BacktestCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		backtest(x_test, x_f_test, y_next, t_test, model, subset = False)
		if DEMO == False:
			sub_xt, sub_xf, sub_y, sub_t = get_subset(x_test, x_f_test, y_next, t_test, ticker_subset)
			backtest(sub_xt, sub_xf, sub_y, sub_t, model, subset=True)

# If we're just loading rather than training, we want to see the performance of the
# experimental model vs. the baseline model.
if load == True:
	# Dictionaries for return logging
	model_returns = {}
	sp_returns = {}
	individual_dates = {}
	individual_model_ret_b = {}
	individual_model_ret_e = {}
	individual_ret = {}

	for logdir in ['./baseline/', './experimental/']:
		model = tf.keras.models.load_model(logdir + 'model.ckpt', custom_objects={'weighted_crossentropy': weighted_crossentropy, 'acc': acc})
		# Display information about the model and the data
		print(model.summary())
		print("Target horizon: {} trading day(s)".format(Y_HORIZON))
		print("Underlying distribution - train: {:.2f}% positive, {:.2f}% negative".format(np.mean(y_train > 0) * 100, ((1 - np.mean(y_train > 0)) * 100)))
		print("Underlying distribution - test:  {:.2f}% positive, {:.2f}% negative".format(np.mean(y_test > 0) * 100, ((1 - np.mean(y_test > 0)) * 100)))

		print('Classes: {}\nTechnicals shape: {}\nFundamentals shape: {}'.format(
			2, x_train.shape[1:], x_f_train.shape[1:]))

		if DEMO:
			print()
			print("NOTE: THIS IS NOT REAL DATA - RANDOMLY GENERATED DUMMY DATA FOR THE PURPOSES OF THIS DEMO")
			print("The model does not work on this data but it is provided to show the flow of the prediction process.")
			print()
		# Backtesting various permutations of the data
		print('Test set return:')
		test_dates, model_returns[logdir], dji_return = backtest(x_test, x_f_test, y_next, t_test, model, subset = False)
		print('Test set accuracy:')
		model.evaluate([x_test, x_f_test], y_test)

		if not DEMO:
			sub_xt, sub_xf, sub_y, sub_t = get_subset(x_test, x_f_test, y_next, t_test, ticker_subset)
			print('S&P500 return:')
			_, sp_returns[logdir], _ = backtest(sub_xt, sub_xf, sub_y, sub_t, model, subset=True)
			print('S&P500 accuracy:')
			model.evaluate([sub_xt, sub_xf], sub_y)

			# Backtest on specific tickers to see results
			sample_tickers = ['INTC', 'MSFT', 'NVDA', 'KO', 'DIS']
			for tick in sample_tickers:
				sub_xt, sub_xf, sub_y, sub_t = get_subset(x_test, x_f_test, y_next, t_test, [tick])
				if logdir == './baseline/':
					individual_dates[tick], individual_model_ret_b[tick], individual_ret[tick] = backtest(sub_xt, sub_xf, sub_y, sub_t, model, subset=True)
				else:
					individual_dates[tick], individual_model_ret_e[tick], individual_ret[tick] = backtest(sub_xt, sub_xf, sub_y, sub_t, model, subset=True)

	# Log some specific return information  for charting results
	path = 'plotting.h5'
	hf = h5py.File(path, 'w')

	# Save model return information
	pd.DataFrame(test_dates).to_hdf(path, key='test_dates')
	hf.create_dataset('dji_ret', data=dji_return)
	hf.create_dataset('model_ret_b', data=model_returns['./baseline/'])
	hf.create_dataset('model_ret_e', data=model_returns['./experimental/'])

	for tick in sample_tickers:
		pd.DataFrame(individual_dates[tick]).to_hdf(path, key=tick+'_dates')
		hf.create_dataset(tick + '_ret', data=individual_ret[tick])
		hf.create_dataset(tick + '_ret_b', data=individual_model_ret_b[tick])
		hf.create_dataset(tick + '_ret_e', data=individual_model_ret_e[tick])

	hf.close()

else:
	print('Constructing model...', end='')
	sys.stdout.flush()
	# Build the model!
	model = Model(input_shape_A=x_train.shape[1:], input_shape_B=x_f_train.shape[1:],
		t_steps=T_STEPS, f_steps=F_STEPS, n_y=len(np.unique(y_train)))
	print('done.')
	model = model.model

	# Tensorboard logging information
	print('Building session graph and setting up Tensorboard logging...')
	current_time = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
	log_dir = 'logs/' + current_time

	sess = tf.Session()
	summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
	print('Done.')

	# Display information about the model and the data
	print(model.summary())
	print("Target horizon: {} trading day(s)".format(Y_HORIZON))
	print("Underlying distribution - train: {:.2f}% positive, {:.2f}% negative".format(np.mean(y_train > 0) * 100, ((1 - np.mean(y_train > 0)) * 100)))
	print("Underlying distribution - test:  {:.2f}% positive, {:.2f}% negative".format(np.mean(y_test > 0) * 100, ((1 - np.mean(y_test > 0)) * 100)))

	print('Classes: {}\nTechnicals shape: {}\nFundamentals shape: {}'.format(
		2, x_train.shape[1:], x_f_train.shape[1:]))

	if DEMO:
		print()
		print("NOTE: THIS IS NOT REAL DATA - RANDOMLY GENERATED DUMMY DATA FOR THE PURPOSES OF THIS DEMO")
		print('This dummy data is linearly separable.')
		print()

	# Tensorboard callback for logging purposes
	tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
	                            write_graph=True)
	# Checkpoint callback for saving model
	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir + 'model.ckpt',
		monitor='val_loss', save_best_only=True, verbose=True)

	# Finally, we train the model. We call it with LearningRateSchedular, TensorBoard,
	# save checkpoint, and backtesting callbacks.
	model.fit([x_train, x_f_train], y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
		validation_data=([x_test, x_f_test], y_test),
		callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
		tb_callback, checkpoint_callback, BacktestCallback()])
