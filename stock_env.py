import pandas as pd
import numpy as np
import random

# Stock trading environment. Modeled after Gym environments.
class StockEnvironment(object):
	
	def __init__(self, path, random = False):
		self.action_space = ['LONG', 'OUT', 'SHORT'] # possible actions
		self.random = random
		if self.random:
			print("Populating with random data because of lack of access to dataset.")
			print("Unlikely to learn a meaningful function.")
			self.X = pd.DataFrame(np.random.randn(100000, 232))
			self.tickers = (np.random.rand(100000) * 500).astype(int)
		else:
			self.data = pd.HDFStore(path) # load
			self.X = self.data.select('X_extras2')
			mask = self.X.marketcap > 1000 # Ignore very small companies

			self.X = self.X[mask]
			self.X = self.X.reset_index(drop=True)

			# Drop tickers corresponding to dropped companies
			self.tickers = self.data.select('ticks')
			self.tickers = self.tickers[mask]
			self.tickers = self.tickers.reset_index(drop=True)

			self.X = self.X.drop(['permaticker', 'closeunadj_^DJI'], axis=1)

			# Normalize to unit variance and zero mean.
			self.X = (self.X - np.mean(self.X, axis=0)) / (np.std(self.X, axis=0) + 1e-10)

		self.reset()

	def reset(self):
		self.curr_ticker=[]
		# Find a ticker with at least 100 samples
		while len(self.curr_ticker) < 100:
			choice = random.choice(self.tickers)
			self.curr_ticker = self.X[self.tickers == choice]
		self.curr_ticker = self.X[self.tickers == choice].copy()
		self.idx = random.choice(range(len(self.curr_ticker) - 100))
		self.start_idx = self.idx

		# For bookkeeping
		self.buys = 0
		self.shorts = 0
		self.outs = 0

		self.max_idx = self.idx + 100
		self.position = 0
		if not self.random:
			self.start_price = self.curr_ticker.iloc[0].close
		else:
			self.start_price = self.curr_ticker.iloc[0][0]
		self.last_price = self.start_price

	def step(self, action):
		done = False
		self.idx += 1

		if self.idx >= self.max_idx:
			done = True
			reward = 0
		else:
			if not self.random:
				price = self.curr_ticker.iloc[self.idx].close
			else:
				price = self.curr_ticker.iloc[self.idx][0]
			if self.action_space[action] == 'LONG':
				self.position = 1
				self.buys += 1
			elif self.action_space[action] == 'SHORT':
				self.position = -1
				self.shorts += 1
			elif self.action_space[action] == 'OUT':
				self.position = 0
				self.outs += 1

			# If position lines up with price movement, reward = 1.
			# If opposite to price movement, reward = -1.
			# Else, reward = 0.
			reward = int(price > self.last_price) * self.position
			self.last_price = price

		return np.array(self.curr_ticker.iloc[self.idx]), reward, done, {}