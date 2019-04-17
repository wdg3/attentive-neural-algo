import tensorflow as tf
import matplotlib.pyplot as plt

from gp_curves_reader import *
from attentive_np import Attention, LatentModel

def plot_functions(target_x, target_y, context_x, context_y, pred_y, std):

	plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
	plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
	plt.plot(context_x[0], context_y[0], 'ko', markersize=6)
	plt.fill_between(
		target_x[0, :, 0],
		pred_y[0, :, 0] - std[0, :, 0],
		pred_y[0, :, 0] + std[0, :, 0],
		alpha=0.2,
		facecolor='#65c9f7',
		interpolate=True)

	plt.yticks([-2, 0, 2], fontsize=16)
	plt.xticks([-2, 0, 2], fontsize=16)
	plt.ylim([-2, 2])
	plt.grid('off')
	ax = plt.gca()
	plt.show()

def main():

	TRAINING_ITERATIONS = 100000 #@param {type:"number"}
	MAX_CONTEXT_POINTS = 20 #@param {type:"number"}
	PLOT_AFTER = 10000 #@param {type:"number"}
	HIDDEN_SIZE = 128 #@param {type:"number"}
	MODEL_TYPE = 'ANP' #@param ['NP','ANP']
	ATTENTION_TYPE = 'multihead' #@param ['uniform','laplace','dot_product','multihead']
	random_kernel_parameters = True #@param {type:"boolean"}

	tf.logging.set_verbosity(tf.logging.ERROR)

	tf.reset_default_graph()
	dataset_train = GPCurvesReader(
		batch_size=16, max_num_context=MAX_CONTEXT_POINTS)
	data_train = dataset_train.generate_curves()

	dataset_test = GPCurvesReader(
		batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)
	data_test = dataset_test.generate_curves()

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

	_, _, log_prob, _, loss = model(data_train.query, data_train.num_total_points,
									data_train.target_y)

	mu, sigma, _, _, _ = model(data_test.query, data_test.num_total_points)

	optimizer = tf.train.AdamOptimizer(1e-4)
	train_step = optimizer.minimize(loss)
	init = tf.initialize_all_variables()

	with tf.train.MonitoredSession() as sess:
		sess.run(init)

		for it in range(TRAINING_ITERATIONS):
			sess.run([train_step])

			if it % PLOT_AFTER == 0:
				loss_value, pred_y, std_y, target_y, whole_query = sess.run(
					[loss, mu, sigma, data_test.target_y, data_test.query])

				(context_x, context_y), target_x = whole_query
				print("Iteration: {}, loss: {}".format(it, loss_value))

				plot_functions(target_x, target_y, context_x, context_y, pred_y, std_y)

if __name__ == "__main__":
	main()