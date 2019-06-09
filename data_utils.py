import pandas as pd
import numpy as np
import h5py
from datetime import datetime

# This is a utility function used to construct the properly-shaped dataset
# from our raw Quandl datasets.
def build_dataset(path, t_steps, f_steps):
	X_store = pd.HDFStore(path + 'x.h5')
	date_store = pd.HDFStore(path + 'dates.h5')
	Y_store = pd.HDFStore(path + 'y.h5')
	
	technicals = X_store.select('X_t')
	fundamentals = X_store.select('X_f')
	dates = date_store.select('dates')
	Y = Y_store.select('Y')

	X_store.close()
	date_store.close()
	Y_store.close()
	Y, X_t, X_f, D = frame_to_time_series(technicals, Y, fundamentals, dates, t_steps, f_steps)

	path = path + '/data.h5'
	hf = h5py.File(path, 'w')

	pd.DataFrame(D).to_hdf(path, key='dates')
	hf.create_dataset('X_t', data=X_t)
	hf.create_dataset('X_f', data=X_f)
	hf.create_dataset('Y', data=Y)

	hf.close()

# Constructs fundamental and technical time series of the appropriate shape
# (m x 8 x 210), (m x 20 x 23) from pandas dataframes built from the Quandl data.
def frame_to_time_series(X_in, Y_in, F_in, dates_in, t_steps, f_steps):
	skip_step = 4 # Every 4 trading days for memory reasons

	F_dates_ticks = F_in[['ticker', 'datekey']]
	F_in = F_in.drop(['ticker', 'datekey'], axis=1)

	X_t = np.zeros((4000000, t_steps, X_in.shape[-1]))
	X_f = np.zeros((4000000, f_steps, F_in.shape[-1]))
	Y = np.zeros((4000000, Y_in.shape[-1]))
	D = np.zeros((4000000, dates_in.shape[-1]), dtype=object)

	# A lot going on in this loop. Look up the datekey corresponding to
	# Quarterly income statements for a stock-date pairing.
	# Then, if we have enoough backward-looking quarters and trading days,
	# add those sequences to their respective datasets.
	k = 0
	for i in range(t_steps, len(X_in), skip_step):
		d = dates_in.iloc[i]
		ticker = d.ticker
		date = d.date
		key = d.datekey
		if (float(date[:4])-float(key[:4])) <= 1 and ticker == dates_in.iloc[i-t_steps].ticker:
			# Filtering operation to find the write fundamental data
			f = F_in[(ticker==F_dates_ticks.ticker)&(key==F_dates_ticks.datekey)]
			if len(f) == 1:
				f_index = f.index[0]
				if F_dates_ticks.iloc[f_index].ticker == F_dates_ticks.iloc[f_index - f_steps].ticker:
					X_t[k, :, :] = X_in.iloc[i-t_steps+1:i+1]
					X_f[k, :, :] = F_in.iloc[f_index-f_steps+1:f_index+1]
					Y[k, :] = Y_in.iloc[i]
					D[k, :] = dates_in.iloc[i]
					k += 1

	Y = Y[:k]
	X_t = X_t[:k]
	X_f = X_f[:k]
	D = D[:k]

	return Y, X_t, X_f, D

# Load our dataset for training.
def load_dataset(path, y_horizon):
	# We save Y labels corresponding to several time horizons and select
	# the one we want at training time.
	# Limit to 1,000,000 examples for memory reasons.
	Y_HORIZONS = [1, 10, 128, 20, 256, 32, 5, 64]
	store = h5py.File(path + 'data.h5', 'r')
	X_t   = np.array(store.get('X_t'))
	X_t   = X_t[-1000000:]
	X_f   = np.array(store.get('X_f'))
	X_f   = X_f[-1000000:]
	Y     = np.array(store.get('Y'))
	Y     = Y[-1000000:]
	store.close()
	dates = np.array(pd.HDFStore(path + 'data.h5').select('dates'))
	dates = dates[-1000000:]

	# Label lookup. Y_next can be different depending on how backtested
	# return is implemented.
	Y_next = Y[:, Y_HORIZONS.index(32)]
	Y = Y[:,Y_HORIZONS.index(y_horizon)]

	valid_mask = Y != -1
	X_t = X_t[valid_mask]
	X_f = X_f[valid_mask]
	Y = Y[valid_mask]
	Y_next = Y_next[valid_mask]
	dates = dates[valid_mask]

	# Test set corresponds to all datapoints after a certain data.
	test_date = datetime(2018, 7, 1)
	train_mask = np.array(pd.Series(dates[:,1]).apply(lambda x: datetime.strptime(x, '%Y-%m-%d') < test_date))
	test_mask = ~train_mask

	return ((X_t[train_mask], Y[train_mask]),
			(X_t[test_mask], Y[test_mask]),
			(X_f[train_mask], X_f[test_mask]),
			(dates[train_mask], dates[test_mask]),
			Y_next[test_mask])

if __name__ == '__main__':
	build_dataset('../data/', 20, 8)