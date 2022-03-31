# utils.py
# A module containing utilities for SRGAN including VGG based loss and
# image preprocessing.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19


class VGGLoss:
	def __init__(self, hr_shape):
		# Initialize VGG19 model.
		vgg = VGG19(
			weights="imagenet", include_top=False, input_shape=hr_shape
		)

		# Only use the top X layers (see references for the reason why)
		self.vgg = Model(
			inputs=vgg.inputs, outputs=vgg.layers[10].output, 
			name="vgg19"
		)
		self.vgg.summary()
		self.vgg.trainable = False

		# Initialize mean squared error loss.
		self.mse = keras.losses.MeanSquaredError(name="vgg_mse")


	def compute_loss(self, y_true, y_pred):
		# Pass both the real and fake (generated) high res images
		# through the VGG19 model.
		real_features = self.vgg(y_true)
		fake_features = self.vgg(y_pred)

		# Return the MSE between the real and generated images.
		return self.mse(real_features, fake_features)


def resize_images(images):
	# Reshape lr & hr images to the set sizes. Since we're using the
	# bicubic x4 subset, make sure that the hr dimensions are 4x the lr
	# dimensions (hr = (128, 128, 3), lr = (32, 32, 3)).
	hr_dims = [128, 128]
	lr_dims = [32, 32]
	return {
		"hr": tf.image.resize(images["hr"], hr_dims, method="bicubic"),
		"lr": tf.image.resize(images["lr"], lr_dims, method="bicubic"),
	}


def scale_images(images):
	# Normalize (scale) values (divide by 255.0).
	return {
		"hr": images["hr"] / 255.0, 
		"lr": images["lr"] / 255.0
	}