import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import trfl
import gym

from attentive_np.cartpole_attentive_np import *
from rl_models.ddqn_anp import *
from cartpole_data import CartpoleData

from rl_models.ddqn import Memory

env = gym.make('LunarLander-v2')

state_size  = 8
action_size = 4

train_episodes = 1000
max_steps = 1000
gamma = 0.99

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

memory_size = 10000
batch_size  = 64
context_size = 64
pretrain_length = batch_size

hidden_size = 32
learning_rate = 1e-4

update_target_every = 1000

HIDDEN_SIZE = 32#@param {type:"number"}
MODEL_TYPE = 'ANP' #@param ['NP','ANP']
ATTENTION_TYPE = 'multihead' #@param ['uniform','laplace','dot_product','multihead']

latent_encoder_output_sizes = [HIDDEN_SIZE] * 4
num_latents = HIDDEN_SIZE
deterministic_encoder_output_sizes= [HIDDEN_SIZE] * 4
decoder_output_sizes = [HIDDEN_SIZE] * 2 + [action_size]
use_deterministic_path = True

if MODEL_TYPE == "ANP":
	attention = Attention(rep="mlp", output_sizes=[HIDDEN_SIZE] * 2,
						  att_type="multihead")
elif MODEL_TYPE == "NP":
	attention = Attention(rep="identity", output_sizes=None, att_type="uniform")
else:
	raise NameError("MODEL_TYPE not among ['ANP', 'NP']")


model = LatentModel(latent_encoder_output_sizes, num_latents,
					decoder_output_sizes, use_deterministic_path,
					deterministic_encoder_output_sizes, attention)

tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

mainQN = QNetwork(name='main_qn', model=model, state_size=state_size, action_size=action_size,
				  hidden_size=hidden_size, learning_rate=learning_rate,
				  batch_size=batch_size, context_size=context_size)
targetQN = QNetwork(name='target_qn', model=model, state_size=state_size, action_size=action_size,
				  hidden_size=hidden_size, learning_rate=learning_rate,
				  batch_size=batch_size, context_size=context_size)

env.reset()

data = CartpoleData(batch_size=batch_size,
					max_num_context=context_size,
					random_num_context=False,
					x_size=state_size,
					y_size=action_size,
					testing=False)

state, reward, done, _ = env.step(env.action_space.sample())

memory = Memory(max_size=memory_size)

for i in range(pretrain_length):
	action = env.action_space.sample()
	next_state, reward, done, _ = env.step(action)
	if done:
		next_state = np.zeros(state.shape)
		memory.add((state, action, reward, next_state))

		env.reset()
		state, reward, done, _ = env.step(env.action_space.sample())
	else:
		memory.add((state, action, reward, next_state))

rewards_list = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	step = 0
	for ep in range(1, train_episodes):
		total_reward = 0
		t = 0
		(context_x, context_y) = data.generate_context(memory)
		while not done:
			step += 1
			env.render()

			if step % update_target_every == 0:
				copy_model_parameters(sess, mainQN, targetQN)
				print("\nCopied model parameters to target network.")

			explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * step)
			if explore_p > np.random.rand():
				action = env.action_space.sample()
			else:
				new_state = state.reshape((1, *state.shape))
				new_state = np.tile(new_state, batch_size)

				feed = {mainQN._context_x: context_x.eval(),
						mainQN._context_y: context_y.eval(),
						mainQN._target_x: new_state.reshape((batch_size, 1, state_size))}
				Qs = sess.run(mainQN.output, feed_dict=feed)
				Qs = np.mean(Qs, axis=0)
				action = np.argmax(Qs)

			next_state, reward, done, _ = env.step(action)

			total_reward += reward

			memory.add((state, action, reward, next_state))
			state = next_state
			t += 1

			batch = memory.sample(batch_size)
			states = np.array([each[0] for each in batch])
			actions = np.array([each[1] for each in batch])
			rewards = np.array([each[2] for each in batch])
			next_states = np.array([each[3] for each in batch])

			target_Qs = sess.run(targetQN.output, feed_dict={targetQN._context_x: context_x.eval(),
															 targetQN._context_y: context_y.eval(),
															 targetQN._target_x: next_states.reshape(batch_size, 1, state_size)})

			episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
			target_Qs[episode_ends] = (0, 0, 0, 0)

			loss, _ = sess.run([mainQN.loss, mainQN.opt],
								feed_dict={mainQN._context_x: context_x.eval(),
								mainQN._context_y: context_y.eval(),
								mainQN._target_x: states.reshape(batch_size, 1, state_size),
								mainQN._targetQs: target_Qs,
								mainQN.reward: rewards,
								mainQN._actions: actions})

		next_state = np.zeros(state.shape)

		print('Episode: {}'.format(ep),
			  'Total reward: {}'.format(total_reward),
			  'Training loss: {:.4f}'.format(loss),
			  'Explore P: {:.4f}'.format(explore_p))
		rewards_list.append((ep, total_reward))

		memory.add((state, action, reward, next_state))

		env.reset()

		state, reward, done, _ = env.step(env.action_space.sample())

	def running_mean(x, N):
		cumsum = np.cumsum(np.inster(x, 0, 0))
		return (cumsum[N:] - cumsum[:-N]) / N

	eps, rews = np.array(reward_list).T
	smoothed_rews = running_mean(rews, 10)
	plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
	plt.plot(eps, rews, color='grey', alpha=0.3)
	plt.xlabel('Episode')
	plt.ylabel('Total Reward')
	plt.show()

print(context)