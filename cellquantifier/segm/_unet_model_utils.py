import keras.backend as k
import tensorflow as tf
from tensorflow.compat.v1.nn import softmax_cross_entropy_with_logits_v2

def channel_precision(channel, name):

	"""
	Wraps the channel precision metric to evaluate the segmentation
	precision for different channels such as interior and boundary

	Pseudo code
	----------
	1. Define precision_func
	2.

	Parameters
	----------
	channel: 2D ndarray,
		A single channel from a multi-channel tensor. Each channel may
		represent the interior, boundary, etc.

	name: str,
		A name to assign to the metric e.g 'boundary_precision'

	y_true: ndarray,
		The true segmentation result

	y_pred: ndarray,
		The segmentation result predicted by the network

	"""

	def precision_func(y_true, y_pred):

		y_pred_tmp = k.cast(tf.equal(k.argmax(y_pred, axis=-1), channel), "float32")
		true_positives = k.sum(k.round(k.clip(y_true[:,:,:,channel] * y_pred_tmp, 0, 1)))
		predicted_positives = k.sum(k.round(k.clip(y_pred_tmp, 0, 1)))
		precision = true_positives / (predicted_positives + k.epsilon())

		return precision

	precision_func.__name__ = name

	return precision_func


def channel_recall(channel, name):

	"""
	Wraps the channel recall metric to evaluate the segmentation
	recall for different channels such as interior and boundary

	Pseudo code
	----------
	1. Define precision_func
	2.

	Parameters
	----------
	channel: 2D ndarray,
		A single channel from a multi-channel tensor. Each channel may
		represent the interior, boundary, etc.

	name: str,
		A name to assign to the metric e.g 'boundary_precision'

	y_true: ndarray,
		The true segmentation result

	y_pred: ndarray,
		The segmentation result predicted by the network

	"""

	def recall_func(y_true, y_pred):

		y_pred_tmp = k.cast(tf.equal( k.argmax(y_pred, axis=-1), channel), "float32")
		true_positives = k.sum(k.round(k.clip(y_true[:,:,:,channel] * y_pred_tmp, 0, 1)))
		possible_positives = k.sum(k.round(k.clip(y_true[:,:,:,channel], 0, 1)))
		recall = true_positives / (possible_positives + k.epsilon())

		return recall

	recall_func.__name__ = name

	return recall_func

def weighted_crossentropy(y_true, y_pred):

	"""
	Defines the weighted_crossentropy loss function

	Pseudo code
	----------
	1. Define precision_func
	2.

	Parameters
	----------

	y_true: ndarray,
		The true segmentation result

	y_pred: ndarray,
		The segmentation result predicted by the network

	"""

	class_weights = tf.constant([[[[1., 1., 10.]]]])
	unweighted_losses = softmax_cross_entropy_with_logits_v2(labels=y_true,
															 logits=y_pred)

	weights = tf.reduce_sum(class_weights*y_true, axis=-1)
	weighted_losses = weights*unweighted_losses
	loss = tf.reduce_mean(weighted_losses)

	return loss
