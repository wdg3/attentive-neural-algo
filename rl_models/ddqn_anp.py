import tensorflow as tf
import numpy as np
import trfl, gym

from collections import deque
from attentive_np.cartpole_attentive_np import Attention, LatentModel
from attentive_np.cartpole_reader import *

class QNetwork(object):

	def __init__(self, name, learning_rate=0.01, state_size=4,
				 action_size=2, hidden_size=10, batch_size=20):

		with tf.variable_scope(name):
			HIDDEN_SIZE = 128 #@param {type:"number"}
			MODEL_TYPE = 'ANP' #@param ['NP','ANP']
			ATTENTION_TYPE = 'multihead' #@param ['uniform','laplace','dot_product','multihead']

			latent_encoder_output_sizes = [HIDDEN_SIZE] * 4
			num_latents = HIDDEN_SIZE
			deterministic_encoder_output_sizes= [HIDDEN_SIZE] * 4
			decoder_output_sizes = [HIDDEN_SIZE] * 2 + [2]
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

			self._inputs = tf.placeholder(tf.float32, [batch_size, None, state_size],
						   name='inputs')

			self._actions = tf.placeholder(tf.int32, [batch_size], name='actions')

			self._context_xs = tf.placeholder(tf.float32, [batch_size, None, state_size], name='x_context')
			self._context_ys = tf.placeholder(tf.float32, [batch_size, None, 2], name='y_context')
			self._query = ((self._context_xs, self._context_ys), self._inputs)

			self.output, _, _, _  = model(self._query, num_targets=2)
			self.output = tf.squeeze(self.output)
			print(self.output.shape)

			self.name = name

			self._targetQs = tf.placeholder(tf.float32, [batch_size, action_size], name='target')
			self.reward = tf.placeholder(tf.float32, [batch_size], name='reward')
			self.discount = tf.constant(0.99, shape=[batch_size], dtype=tf.float32, name='discount')

			#print(self.output)
			#print(self._actions)
			#print(self._targetQs)

			q_loss, q_learning = trfl.double_qlearning(self.output, self._actions, self.reward,
													   self.discount, self._targetQs, self.output)
			self.loss = tf.reduce_mean(q_loss)
			self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

class Memory(object):

	def __init__(self, max_size=1000):

		self.buffer = deque(maxlen=max_size)

	def add(self, experience):
			self.buffer.append(experience)

	def sample(self, batch_size):
			idx = np.random.choice(np.arange(len(self.buffer)),
								   size=batch_size,
								   replace=False)

			return [self.buffer[ii] for ii in idx]

def copy_model_parameters(sess, estimator1, estimator2):

	e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.name)]
	e1_params = sorted(e1_params, key=lambda v: v.name)
	e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.name)]
	e2_params = sorted(e2_params, key=lambda v: v.name)

	update_ops = []
	for e1_v, e2_v in zip(e1_params, e2_params):
		op = e2_v.assign(e1_v)
		update_ops.append(op)

	sess.run(update_ops)
