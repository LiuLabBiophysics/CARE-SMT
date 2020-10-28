from ._unet_data_utils import *
from ._unet_model import *
from ._unet_model_utils import *
from ._unet_vis import *
from skimage.morphology import *
from skimage.measure import label
import keras
import pandas as pd

def train_model(train_gen,
				valid_gen,
				crop_size,
				save_dir=None,
				metrics=None,
				optimizer=None,
				loss=None,
				epochs=10,
				steps_per_epoch=None,
				callbacks=1,
				verbose=1):

	"""
	Builds a new model that can be used to segment images

	Pseudo code
	----------
	1. Check if metrics have been specified
	2. Check if optimizer has been specified
	3. Partition files into training and testing
	4. Instantiate a U-Net model
	5. Train the U-Net model
	6. Display statistics

	Parameters
	----------

	input_dir: str
		directory containing the training and testing images
	target_dir: str
		directory containing the training and testing masks
	crop_size: str
		the size of the patch extracted from each image,mask
	metrics: list
		a list of keras metrics used to judge model performance
	optimizer: keras optimizer object
		optimization method e.g. gradient descent
	loss: keras loss-function object
		mathematical loss-function to be minimized during training
	batch_size: int
		number of images to train on in each epoch
	epochs: int
		number of epochs to train for
	fraction_validation: float
		fraction of training data to use for validation
	callbacks: str,
	 utilities called at certain points during model trainin
	verbose: int
		verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch

	"""

	# """
	# ~~~~~~~~~~Fill default params~~~~~~~~~~~~~~
	# """

	if not metrics:
		metrics = [keras.metrics.categorical_accuracy,
				   channel_recall(channel=0, name="background_recall"),
				   channel_precision(channel=0, name="background_precision"),
				   channel_recall(channel=1, name="interior_recall"),
				   channel_precision(channel=1, name="interior_precision"),
				   channel_recall(channel=2, name="boundary_recall"),
				   channel_precision(channel=2, name="boundary_precision"),
				  ]

	if not optimizer:
			optimizer = keras.optimizers.RMSprop(lr=1e-4)


	# """
	# ~~~~~~~~~~Train the u-net model~~~~~~~~~~~~~~
	# """

	model = unet_model(input_size=crop_size); model.summary()
	model.compile(loss=weighted_crossentropy, metrics=metrics, optimizer=optimizer)

	statistics = model.fit(x=train_gen, epochs=epochs,
						   steps_per_epoch=steps_per_epoch,
						   validation_data=valid_gen)

	stats_df = pd.DataFrame(statistics.history)

	if save_dir:
		stats_df.to_csv(save_dir + '/train_stats.csv')
		model.save(save_dir + '/model.h5')



def make_prediction(stack, model, output_path):

	"""
	Given an image and a model, makes a prediction for the segmentation
	mask

	Pseudo code
	----------

	Parameters
	----------
	im_arr: list
		a stack of images to make predictions on
	model: model weights
		the weights to plugin to model generate by unet_model()

	"""

	# """
	# ~~~~~~~~~~Check input and reshape~~~~~~~~~~~~~~
	# """

	if len(stack.shape) < 3:
		stack = stack.reshape((1,) + stack.shape)

	stack = stack.reshape(stack.shape + (1,))
	prediction = model.predict(stack, batch_size=1)

	# """
	# ~~~~~~~~~~Transform prediction to label matrices~~~~~~~~
	# """

	for i in range(len(prediction)):
		probmap = prediction[i].squeeze()
		pred = probmap_to_pred(probmap)
		out = pred_to_label(pred)

		# """
		# ~~~~~~~~~~Convert to ImageJ format, output~~~~~~~~
		# """

		out = img_as_ubyte(out); shape = out.shape
		imsave(output_path + '/out%s.tif' % str(i), out)

def probmap_to_pred(probmap, boundary_boost_factor=1):

    pred = np.argmax(probmap * [1, 1, boundary_boost_factor], -1)

    return pred


def pred_to_label(pred, cell_label=1):

    cell = (pred == cell_label)
    [lbl, num] = label(cell, return_num=True)

    return lbl
