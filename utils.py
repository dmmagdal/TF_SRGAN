# utils.py
# A module containing utilities for SRGAN including VGG based loss and
# image preprocessing.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19
from matplotlib import pyplot as plt


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


def save_images(valid_data, generator, e, offset_by_one=False):
	# Randomly sample from validation data and perform super resolution
	# on that sample.
	random.seed(42)
	index = random.randint(0, len(list(valid_data.as_numpy_iterator())))
	sample = list(valid_data.as_numpy_iterator())[index]
	src_img = sample["lr"]
	tar_img = sample["hr"]
	n_dims = len(tf.shape(src_img))

	# If the sample is not in (batch_size, height, width, channel)
	# format, then it must be in (height, width, channel) format.
	if n_dims == 3:
		src_img = tf.expand_dims(src_img, axis=0)
		tar_img = tf.expand_dims(tar_img, axis=0)

	gen_img = generator.predict(src_img) * 255.0

	# Plot all three images.
	plt.figure(figsize=(16, 8))
	plt.subplot(231)
	plt.title("LR Image")
	plt.imshow(src_img[0, :, :, :])
	plt.subplot(232)
	plt.title("Superresolution")
	plt.imshow(gen_img[0, :, :, :])
	plt.subplot(233)
	plt.title("HR Image")
	plt.imshow(tar_img[0, :, :, :])
	if offset_by_one:
		e += 1
	plt.savefig(f"SRGAN_Generator_Sample{e}.png")
	plt.close()