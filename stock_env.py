import pandas as pd
import numpy as np
import random

class StockEnvironment(object):
	
	def __init__(self, state_size, path):
		self.action_space = ['LONG', 'OUT', 'SHORT']
		self.data = pd.HDFStore(path)
		self.X = self.data.select('X_extras2')
		mask = self.X.marketcap > 1000
		self.X = self.X[mask]
		self.X = self.X.reset_index(drop=True)
		self.tickers = self.data.select('ticks')
		self.tickers = self.tickers[mask]
		self.tickers = self.tickers.reset_index(drop=True)
		self.X = self.X.drop(['permaticker', 'closeunadj_^DJI'], axis=1)
		self.X = (self.X - np.mean(self.X, axis=0)) / (np.std(self.X, axis=0) + 1e-10)
		print(self.X.shape)
		self.reset()

	def reset(self):
		choice = []
		while len(choice) < 100:
			choice = random.choice(self.tickers)
		print(choice)
		self.curr_ticker = self.X[self.tickers == choice].copy()
		self.idx = random.choice(range(len(self.curr_ticker) - 100))
		self.start_idx = self.idx
		self.buys = 0
		self.shorts = 0
		self.outs = 0
		self.max_idx = self.idx + 100
		self.position = 0
		self.start_price = self.curr_ticker.iloc[0].close
		self.last_price = self.start_price

	def step(self, action):
		done = False
		self.idx += 1

		if self.idx >= self.max_idx:
			done = True
			reward = 0
		else:
			price = self.curr_ticker.iloc[self.idx].close
			if self.action_space[action] == 'LONG':
				self.position = 1
				self.buys += 1
			elif self.action_space[action] == 'SHORT':
				self.position = -1
				self.shorts += 1
			elif self.action_space[action] == 'OUT':
				self.position = 0
				self.outs += 1
			reward = int(price > self.last_price) * self.position
			self.last_price = price

		return np.array(self.curr_ticker.iloc[self.idx]), reward, done, {}