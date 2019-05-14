import numpy as np
import pandas as pd
import h5py

def construct(path, timesteps, skip_step):
  # Load data from file
  store = pd.HDFStore(path)
  X2 = store.select('X_extras2')
  print(X2.shape)
  Y = np.array(store.select('Y'))  
  print(np.mean(Y))
  print(Y.shape)
  tickers = X2.permaticker.unique()
  store.close()
  X2 = X2.reset_index(drop=True)
  # Collect tickers without enough entries and drop from dataset
  print('Collecting ineligible tickers...')
  drop_tickers = []
  for t in tickers:
    if X2.loc[X2.permaticker == t].shape[0] < (skip_step*timesteps) + 1:
      drop_tickers.append(t)
  print('Creating ticker mask...')
  for t in drop_tickers:
    mask = X2.permaticker != t
    X2 = X2[mask]
    Y = Y[mask]
  print(np.mean(Y))
  print(Y.shape)
  print(X2.shape)
  print(Y[-100:])

  ts = X2.permaticker
  #X2 = X2.drop('permaticker', axis=1)
  print('Normalizing...')
  mu = np.mean(X2, axis=0)
  sigma = np.std(X2, axis=0)
  X2 = (X2 - mu) / (sigma + 1e-10)  

  # Create backward-looking time series for each eligible datapoint
  print('Converting to time series form...')
  xs = np.zeros((X2.shape[0] - (len(ts.unique()) * timesteps * skip_step), timesteps, X2.shape[1]))
  ys = np.zeros((Y.shape[0] - (len(ts.unique()) * timesteps) * skip_step, 1))
  ticks = np.zeros((Y.shape[0] - (len(ts.unique()) * timesteps) * skip_step))
  k = 0
  for t in ts.unique():
    mask = ts == t
    chonk = X2[mask]
    y_chonk = Y[mask]
    for i in range(timesteps*skip_step, chonk.shape[0]):
      xs[k,:, :] = chonk.iloc[i - (timesteps*skip_step):i:skip_step]
      ys[k,:]    = y_chonk[i]
      ticks[k]   = chonk.iloc[i].permaticker
      k += 1
  print(np.mean(ys))
  print(ys.shape)
  print(ticks)

  print('Saving...')
  # Save full dataset to file
  path = 'xy_' + str(timesteps) + '.h5'
  hf = h5py.File(path, 'w')
  hf.create_dataset('X', data=xs)
  hf.create_dataset('Y', data=ys)
  hf.create_dataset('tickers', data=ticks)

  print(xs.shape)
  print(ys.shape)

  hf.close()

def get_data(path):
  store = h5py.File(path, 'r')
  X     = store.get('X')
  Y     = store.get('Y')
  
  # Shuffle data while preserving X-Y pairings
  #indices = np.random.permutation(X.shape[0])
  print(X.shape)
  print(Y.shape)
  # Train-test split
  #train_i, test_i = indices[:-10000], indices[-10000:]


  X_train = X[:-10000,:,:]
  Y_train = Y[:-10000,:]
  X_test  = X[-10000:,:,:]
  Y_test  = Y[-10000:,:]

  # Check train and test sizes and class distributions
  print('{} training examples, {} test examples'.format(X_train.shape[0], X_test.shape[0]))
  print("Underlying distribution - train: {:.2f}% positive, {:.2f}% negative".format(np.mean(Y_train) * 100, ((1 - np.mean(Y_train)) * 100)))
  print("Underlying distribution - test:  {:.2f}% positive, {:.2f}% negative".format(np.mean(Y_test) * 100, ((1 - np.mean(Y_test)) * 100)))

  return ((X_train, Y_train), (X_test, Y_test))

def main():
  construct('data.h5', 8, 8)

if __name__ == '__main__':
  main()
