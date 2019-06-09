import tensorflow as tf
import numpy as np
import collections

CartpoleRegressionDescription = collections.namedtuple(
	"CartpoleRegressionDescription",
	("query", "target_y", "num_total_points", "num_context_points"))

class CartpoleReader(object):
  """Generates curves using a Gaussian Process (GP).

  Supports vector inputs (x) and vector outputs (y). Kernel is
  mean-squared exponential, using the x-value l2 coordinate distance scaled by
  some factor chosen randomly in a range. Outputs are independent gaussian
  processes.
  """

  def __init__(self,
               batch_size,
               max_num_context,
               x_size=4,
               y_size=2,
               testing=False):
    """Creates a regression dataset of functions sampled from a GP.

    Args:
      batch_size: An integer.
      max_num_context: The max number of observations in the context.
      x_size: Integer >= 1 for length of "x values" vector.
      y_size: Integer >= 1 for length of "y values" vector.
      testing: Boolean that indicates whether we are testing. If so there are
          more targets for visualization.
    """
    self._batch_size = batch_size
    self._max_num_context = max_num_context
    self._x_size = x_size
    self._y_size = y_size
    self._testing = testing

  def generate_curves(self):
    """Builds the op delivering the data.

    Generated functions are `float32` with x values between -2 and 2.
    
    Returns:
      A `CNPRegressionDescription` namedtuple.
    """
    num_context = tf.random_uniform(
        shape=[], minval=3, maxval=self._max_num_context, dtype=tf.int32)

    # If we are testing we want to have more targets and have them evenly
    # distributed in order to plot the function.
    # During training the number of target points and their x-positions are
	# selected at random
    if self._testing:
      num_target = 400
      num_total_points = num_target
      x_values = tf.cast(tf.random_uniform(
          [self._batch_size, num_total_points, self._x_size], 0, 1), tf.float32)
    # During training the number of target points and their x-positions are
    # selected at random
    else:
      num_target = tf.random_uniform(shape=(), minval=0, 
                                     maxval=self._max_num_context - num_context,
                                     dtype=tf.int32)
      num_total_points = num_context + num_target

      x_values = tf.cast(tf.random_uniform(
          [self._batch_size, num_total_points, self._x_size], 0, 1), tf.float32)

    # Sample a curve
    # [batch_size, y_size, num_total_points, 1]
    sums = tf.expand_dims(tf.reduce_sum(x_values, axis=2), axis=-1)
    x0 = tf.expand_dims(x_values[:, :, 0], axis=-1)
    x1 = tf.expand_dims(x_values[:, :, 1], axis=-1)
    x2 = tf.expand_dims(x_values[:, :, 2], axis=-1)
    x3 = tf.expand_dims(x_values[:, :, 3], axis=-1)
    z = tf.sqrt(x0 * x2) > 0.5
    w = (x1 + x3) / x0 > 1
    y_values = tf.cast(tf.cast(z | w, tf.int32), tf.float32)


    # [batch_size, num_total_points, y_size]

    if self._testing:
      # Select the targets
      target_x = x_values
      target_y = y_values

      # Select the observations
      idx = tf.random_shuffle(tf.range(num_target))
      context_x = tf.gather(x_values, idx[:num_context], axis=1)
      context_y = tf.gather(y_values, idx[:num_context], axis=1)

    else:
      # Select the targets which will consist of the context points as well as
      # some new target points
      target_x = x_values[:, :num_target + num_context, :]
      target_y = y_values[:, :num_target + num_context, :]

      # Select the observations
      context_x = x_values[:, :num_context, :]
      context_y = y_values[:, :num_context, :]

    query = ((context_x, context_y), target_x)

    return CartpoleRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=tf.shape(target_x)[1],
        num_context_points=num_context)