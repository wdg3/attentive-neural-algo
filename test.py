import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

from rl_models.ddqn import QNetwork, Memory, copy_model_parameters

env = gym.make('LunarLander-v2')

def main():
	state_size = 8
	action_size = 4

	train_episodes = 10000
	max_steps = 255
	gamma = 0.99

	explore_start = 1.0
	explore_stop = 0.01
	decay_rate = 0.0001

	hidden_size = 32
	learning_rate = 1e-4

	memory_size = 10000
	batch_size = 64
	pretrain_length = batch_size

	update_target_every = 10000

	tf.reset_default_graph()
	mainQN = QNetwork(name='main_qn', state_size=state_size, action_size=action_size,
					  hidden_size=hidden_size,
					  learning_rate=learning_rate, batch_size=batch_size)
	targetQN = QNetwork(name='target_qn', state_size=state_size, action_size=action_size,
						hidden_size=hidden_size,
						learning_rate=learning_rate, batch_size=batch_size)


	env.reset()

	state, reward, done, _ = env.step(env.action_space.sample())

	memory = Memory(max_size=memory_size)

	for ii in range(pretrain_length):
		action = env.action_space.sample()
		next_state, reward, done, _ = env.step(action)

		if done:
			next_state = np.zeros(state.shape)
			memory.add((state, action, reward, next_state))

			env.reset()
			state, reward, done, _ = env.step(env.action_space.sample())

		else:
			memory.add((state, action, reward, next_state))
			state = next_state

	rewards_list = []
	with tf.Session() as sess:
		# Initialize variables
		sess.run(tf.global_variables_initializer())
		
		step = 0
		for ep in range(1, train_episodes):
			total_reward = 0
			t = 0
			while not done:
				step += 1
				# Uncomment this next line to watch the training
				env.render() 
				
				#update target q network
				if step % update_target_every == 0:
					copy_model_parameters(sess, mainQN, targetQN)
					print("\nCopied model parameters to target network.")
				
				explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step) 
				if explore_p > np.random.rand():
					# Make a random action
					action = env.action_space.sample()
				else:
					# Get action from Q-network
					feed = {mainQN._inputs: state.reshape((1, *state.shape))}
					Qs = sess.run(mainQN.output, feed_dict=feed)
					action = np.argmax(Qs)

				# Take action, get new state and reward
				next_state, reward, done, _ = env.step(action)
		
				total_reward += reward

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
				#in this example (and in Deep Q Networks) use targetQN for the target values
				#target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})
				target_Qs = sess.run(targetQN.output, feed_dict={targetQN._inputs: next_states})
				
				# Set target_Qs to 0 for states where episode ends
				episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
				target_Qs[episode_ends] = (0, 0, 0, 0)

				#TRFL way, calculate td_error within TRFL
				loss, _ = sess.run([mainQN.loss, mainQN.opt],
									feed_dict={mainQN._inputs: states,
											   mainQN._targetQs: target_Qs,
											   mainQN.reward: rewards,
											   mainQN._actions: actions})

						# the episode ends so no next state
			next_state = np.zeros(state.shape)

			print('Episode: {}'.format(ep),
				  'Total reward: {}'.format(total_reward),
				  'Training loss: {:.4f}'.format(loss),
				  'Explore P: {:.4f}'.format(explore_p))
			rewards_list.append((ep, total_reward))
			
			# Add experience to memory
			memory.add((state, action, reward, next_state))
			
			# Start new episode
			env.reset()
			# Take one random step to get the pole and cart moving
			state, reward, done, _ = env.step(env.action_space.sample())

		def running_mean(x, N):
			cumsum = np.cumsum(np.insert(x, 0, 0))
			return (cumsum[N:] - cumsum[:-N]) / N

		eps, rews = np.array(rewards_list).T
		smoothed_rews = running_mean(rews, 10)
		plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
		plt.plot(eps, rews, color='grey', alpha=0.3)
		plt.xlabel('Episode')
		plt.ylabel('Total Reward')
		plt.show()

if __name__ == '__main__':
	main()



