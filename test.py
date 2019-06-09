import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import random

from rl_models.ddqn_snail import *

from rl_models.ddqn_snail import Memory
from stock_env import StockEnvironment

# Populate random data for demo without dataset
random_data = True

env = StockEnvironment(path='data.h5', random=random_data)

state_size  = 232
action_size = 3

train_episodes = 1000
max_steps = 1000
gamma = 0.99

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

memory_size = 10000
batch_size  = 128
context_size = 32
pretrain_length = batch_size

hidden_size = 128
learning_rate = 1e-4

update_target_every = 1000

# Used only with ANPs
HIDDEN_SIZE = 128#@param {type:"number"}
MODEL_TYPE = 'ANP' #@param ['NP','ANP']
ATTENTION_TYPE = 'multihead' #@param ['uniform','laplace','dot_product','multihead']

# Used only with ANPs
latent_encoder_output_sizes = [HIDDEN_SIZE] * 4
num_latents = HIDDEN_SIZE
deterministic_encoder_output_sizes= [HIDDEN_SIZE] * 4
decoder_output_sizes = [HIDDEN_SIZE] * 2 + [action_size]
use_deterministic_path = True

# Used only with ANPs
if MODEL_TYPE == "ANP":
	attention = Attention(rep="mlp", output_sizes=[HIDDEN_SIZE] * 2,
		att_type="multihead")
elif MODEL_TYPE == "NP":
	attention = Attention(rep="identity", output_sizes=None, att_type="uniform")
else:
	raise NameError("MODEL_TYPE not among ['ANP', 'NP']")

# Used only with ANPs
model = LatentModel(latent_encoder_output_sizes, num_latents,
	decoder_output_sizes, use_deterministic_path,
	deterministic_encoder_output_sizes, attention)

tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

# Define our main deep Q network with TRFL.
mainQN = QNetwork(name='main_qn', state_size=state_size, action_size=action_size,
	hidden_size=hidden_size, learning_rate=learning_rate,
	batch_size=batch_size)

env.reset()
state, reward, done, _ = env.step(random.choice(range(action_size)))

memory = Memory(max_size=memory_size)

# Populate our memory with pretraining session
for i in range(pretrain_length):
	action = random.choice(range(action_size))
	next_state, reward, done, _ = env.step(action)
	if done:
		next_state = np.zeros(state.shape)
		memory.add((state, action, reward, next_state))

		env.reset()
		state, reward, done, _ = env.step(random.choice(range(action_size)))
	else:
		memory.add((state, action, reward, next_state))

rewards_list = []
with tf.Session() as sess:
	# Initialize variables
	sess.run(tf.global_variables_initializer())
	
	step = 0
	for ep in range(1, train_episodes):
		total_reward = 0
		t = 0
		while t < max_steps:
			step += 1
			
			explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step) 
			if explore_p > np.random.rand():
				# Make a random action
				action = random.choice(range(action_size))
			else:
				# Get action from Q-network
				feed = {mainQN._target_x: state.reshape((1, *state.shape))}
				Qs = sess.run(mainQN.output, feed_dict=feed)
				action = np.argmax(Qs)

			# Take action, get new state and reward
			next_state, reward, done, _ = env.step(action)

			total_reward += reward
			
			if done:
				# Show results if the episode is over
				next_state = np.zeros(state.shape)
				t = max_steps
				total_reward = total_reward / (env.idx - env.start_idx)
				print('Episode: {}'.format(ep),
					'Total reward: {}'.format(total_reward),
					'Training loss: {:.4f}'.format(loss),
					'Explore P: {:.4f}'.format(explore_p))
				print('Buys: {:.2f}'.format(env.buys / (env.idx - env.start_idx)),
					'Shorts: {:.2f}'.format(env.shorts / (env.idx - env.start_idx)),
					'Outs: {:.2f}'.format(env.outs / (env.idx - env.start_idx) ))
				rewards_list.append((ep, total_reward))
				
				# Add experience to memory
				memory.add((state, action, reward, next_state))
				
				# Start new episode
				env.reset()
				# Take one random step to get the pole and cart moving
				state, reward, done, _ = env.step(random.choice(range(action_size)))

			else:
				# Add experience to memory
				memory.add((state, action, reward, next_state))
				state = next_state
				t += 1

			# Sample mini-batch from memory
			batch = memory.sample(batch_size)
			states = np.array([each[0] for each in batch])
			actions = np.array([each[1] for each in batch])
			rewards = np.array([each[2] for each in batch])
			next_states = np.array([each[3] for each in batch])
			
			# Train network
			target_Qs = sess.run(mainQN.output, feed_dict={mainQN._target_x: next_states})
			
			# Set target_Qs to 0 for states where episode ends
			episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
			target_Qs[episode_ends] = (0, 0, 0)

			# Get loss from network
			loss, _ = sess.run([mainQN.loss, mainQN.opt],
				feed_dict={mainQN._target_x: states,
				mainQN._targetQs: target_Qs,
				mainQN.reward: rewards,
				mainQN._actions: actions})

			def ewm(x, n):
				cumsum = np.cumsum(np.insert(x, 0, 0))
				return pd.DataFrame(x).ewm(span = n).mean()

	eps, rews = np.array(rewards_list).T
	smoothed_rews = ewm(rews, 20)
	plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
	plt.plot(eps, rews, color='grey', alpha=0.3)
	plt.xlabel('Episode')
	plt.ylabel('Total Reward')
	plt.show()