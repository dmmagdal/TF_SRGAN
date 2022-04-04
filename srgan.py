# srgan.py
# Implement and train the Super Resolution GAN in Tensorflow 2 for
# image super resolution.
# Source: https://www.youtube.com/watch?v=1HqjPqNglPc&t=472s&ab_
# channel=DigitalSreeni
# Source (Github): https://github.com/bnsreenu/python_for_
# microscopists/blob/master/255_256_SRGAN/SRGAN_train.py
# Source (Video Dataset): https://press.liacs.nl/mirflickr/
# mirdownload.html
# Source (Paper): https://arxiv.org/pdf/1609.04802.pdf
# Source (Supplimentary Video): https://www.youtube.com/watch?v=FwvTsx_
# dxn8&ab_channel=AIExpedition
# Source (Another Reference): https://blog.paperspace.com/super-
# resolution-generative-adversarial-networks/
# Source (DGGAN Keras Example): https://keras.io/examples/generative/
# dcgan_overriding_train_step/
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# Blocks to build Generator.
class ResBlock(layers.Layer):
	def __init__(self, **kwargs):
		super().__init__()
		self.conv1 = layers.Conv2D(64, (3, 3), padding="same")
		self.batch_norm1 = layers.BatchNormalization(momentum=0.5)
		self.prelu = layers.PReLU(shared_axes=[1, 2])
		self.conv2 = layers.Conv2D(64, (3, 3), padding="same")
		self.batch_norm2 = layers.BatchNormalization(momentum=0.5)
		self.add = layers.Add()


	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.batch_norm1(x)
		x = self.prelu(x)
		x = self.conv2(x)
		outs = self.batch_norm2(x)
		return self.add([inputs, outs])


	def get_config(self):
		config = super(ResBlock, self).get_config()
		return config


class UpScaleBlock(layers.Layer):
	def __init__(self, **kwargs):
		super().__init__()
		self.conv = layers.Conv2D(256, (3, 3), padding="same")
		self.upsample = layers.UpSampling2D(size=2)
		self.prelu = layers.PReLU(shared_axes=[1, 2])


	def call(self, inputs):
		x = self.conv(inputs)
		x = self.upsample(x)
		outs = self.prelu(x)
		return outs


	def get_config(self):
		config = super(UpScaleBlock, self).get_config()
		return config	


# Generator model.
def create_generator(inputs, num_res_blocks):
	x = layers.Conv2D(64, (9, 9), padding="same")(inputs)
	x = layers.PReLU(shared_axes=[1, 2])(x)

	temp = x # Residual.

	for i in range(num_res_blocks):
		x = ResBlock()(x)

	x = layers.Conv2D(64, (3, 3), padding="same")(x)
	x = layers.BatchNormalization(momentum=0.5)(x)
	x = layers.Add()([x, temp])

	x = UpScaleBlock()(x)
	x = UpScaleBlock()(x)
	
	outputs = layers.Conv2D(3, (9, 9), padding="same")(x)
	
	return Model(inputs=inputs, outputs=outputs, name="generator")


# Block to build Discriminator.
class DiscriminatorBlock(layers.Layer):
	def __init__(self, filters, strides=1, batch_norm=True, **kwargs):
		super().__init__()
		self.conv = layers.Conv2D(
			filters, (3, 3), strides=strides, padding="same"
		)
		self.use_batchnorm = batch_norm
		self.batch_norm = layers.BatchNormalization(momentum=0.8)
		self.leaky_relu = layers.LeakyReLU(alpha=0.2)


	def call(self, inputs):
		x = self.conv(inputs)
		if self.use_batchnorm:
			x = self.batch_norm(x)
		outs = self.leaky_relu(x)
		return outs


	def get_config(self):
		config = super(DiscriminatorBlock, self).get_config()
		config.update({
			"use_batchnorm": self.use_batchnorm,
		})
		return config


# Discriminator model.
def create_discriminator(inputs):
	df = 64
	first_layer = True
	filters = [df, df * 2, df * 2, df * 4, df * 4, df * 8, df * 8]
	strides = [1, 2, 1, 2, 1, 2, 1, 2]

	x = inputs
	for filter, stride in zip(filters, strides):
		if first_layer:
			first_layer = False
		x = DiscriminatorBlock(
			filter, stride, batch_norm=first_layer
		)(x)

	x = layers.Flatten()(x)
	x = layers.Dense(df * 16)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	validity = layers.Dense(1, activation="sigmoid")(x)

	return Model(inputs=inputs, outputs=validity, name="discriminator")


# VGG19.
def build_vgg19(hr_shape):
	vgg = VGG19(
		weights="imagenet", include_top=False, input_shape=hr_shape
	)
	block3_conv4 = 10
	block5_conv4 = 20

	return Model(
		inputs=vgg.inputs, outputs=vgg.layers[block5_conv4].output, 
		name="vgg19"
	)


# Combined model.
def create_combined_model(gen_model, disc_model, vgg_model, lr_inputs, hr_inputs):
	gen_img = gen_model(lr_inputs)

	gen_features = vgg_model(gen_img)

	disc_model.trainable = False
	validity = disc_model(gen_img)

	return Model(
		inputs=[lr_inputs, hr_inputs], outputs=[validity, gen_features], 
		name="gan"
	)


def load_images(image_folder):
	images = []
	for img in os.listdir(image_folder):
		img_res = cv2.imread(os.path.join(image_folder, img))
		img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
		images.append(img_res)
	return images


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
		"lr": images["lr"] / 255.0,
	}


def save_images(valid_data, generator, e):
	# Randomly sample from validation data and perform super resolution
	# on that sample.
	random.seed(42)
	index = random.randint(0, len(list(valid_data.as_numpy_iterator())))
	sample = list(valid_data.as_numpy_iterator())[index]
	src_img = sample["lr"]
	tar_img = sample["hr"]

	# Note: matplotlib pyplot is able to print float values from [0, 1]
	# in their original color (as long as they are floats). In other
	# words, the graphing handles the conversion from grayscale to RGB
	# (scalar multiply by 255). If the values are int from [0,  255],
	# then the graphing is already in RGB and is printed out with no
	# need of conversion (no need to multiply by 255).
	gen_img = generator.predict(src_img) #* 255.0

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
	plt.savefig(f"SRGAN_Generator_Sample{e + 1}.png")
	plt.close()


def main():
	# Load Div2k (bicubic_x4) dataset.
	data = tfds.load("div2k/bicubic_x4", data_dir="./div2k")
	train_data = data["train"]
	valid_data = data["validation"]

	# Reshape lr & hr images to the set sizes. Since we're using the
	# bicubic x4 subset, make sure that the hr dimensions are 4x the lr
	# dimensions (hr = (128, 128, 3), lr = (32, 32, 3)).
	autotune = tf.data.AUTOTUNE
	train_data = train_data.map(
		resize_images, num_parallel_calls=autotune
	)
	valid_data = valid_data.map(
		resize_images, num_parallel_calls=autotune
	)

	# Normalize (scale) values (divide by 255.0).
	train_data = train_data.map(
		scale_images, num_parallel_calls=autotune
	)
	valid_data = valid_data.map(
		scale_images, num_parallel_calls=autotune
	)
	single_sample = list(train_data.as_numpy_iterator())[0]
	hr_shape = tf.shape(single_sample["hr"]).numpy()
	hr_shape = (hr_shape[0], hr_shape[1], hr_shape[2])
	lr_shape = tf.shape(single_sample["lr"]).numpy()
	lr_shape = (lr_shape[0], lr_shape[1], lr_shape[2])
	print(hr_shape)
	print(lr_shape)

	# Define inputs to the models.
	hr_inputs = layers.Input(shape=hr_shape)
	lr_inputs = layers.Input(shape=lr_shape)

	# Initialize models.
	generator = create_generator(lr_inputs, num_res_blocks=16)
	generator.summary()

	discriminator = create_discriminator(hr_inputs)
	disc_opt = keras.optimizers.Adam(beta_1=0.5, beta_2=0.99)
	discriminator.compile(
		loss="binary_crossentropy", 
		# optimizer="adam", 
		optimizer=disc_opt,
		metrics=["accuracy"]
	)
	discriminator.summary()

	vgg = build_vgg19((128, 128, 3))
	vgg.summary()
	vgg.trainable = False

	gan = create_combined_model(
		generator, discriminator, vgg, lr_inputs, hr_inputs
	)
	gan_opt = keras.optimizers.Adam(beta_1=0.5, beta_2=0.99)
	gan.compile(
		loss=["binary_crossentropy", "mse"], 
		loss_weights=[1e-3, 1],
		# optimizer="adam",
		optimizer=gan_opt,
	)
	gan.summary()

	# Training loop.
	epochs = 500#1000#500#100#5
	batch_size = 4
	train_data = train_data.prefetch(buffer_size=autotune)\
		.batch(batch_size)
	valid_data = valid_data.prefetch(buffer_size=autotune)\
		.batch(batch_size)
	for e in range(epochs):
		fake_label = np.zeros((batch_size, 1))
		real_label = np.ones((batch_size, 1))

		g_losses = []
		d_losses = []
		for b in tqdm(train_data):
			hr_imgs = b["hr"]
			lr_imgs = b["lr"]

			fake_imgs = generator.predict_on_batch(lr_imgs)

			# Train discriminator on fake and real HR images.
			discriminator.trainable = True
			d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
			d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)

			# Train generator. Fix discrinimator as non-trainable.
			discriminator.trainable = False

			# Average the discrinimator loss. 
			d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)

			# Extract VGG features (to be used for calculating loss).
			image_features = vgg.predict(hr_imgs)
		
			# Train the image generator via the GAN.
			g_loss, _, _ = gan.train_on_batch(
				[lr_imgs, hr_imgs], [real_label, image_features]
			)

			# Save the loss to a list so that we can average it.
			d_losses.append(d_loss)
			g_losses.append(g_loss)

		# Convert the list of losses to an array to make it easier to
		# average.
		g_losses = np.array(g_losses)
		d_losses = np.array(d_losses)

		# Calculate the average losses for the generator and
		# discriminator.
		g_loss = np.sum(g_losses, axis=0) / len(g_losses)
		d_loss = np.sum(d_losses, axis=0) / len(d_losses)

		# Report the training progress. Save generator after every n
		# epochs.
		print(f"Epoch: {e + 1}, Gen-Loss: {g_loss}, Disc-Loss: {d_loss}")
		if (e + 1) % 10 == 0 or (e + 1) == epochs:
			# generator.save("srgan_generator_epochs" + str(e + 1) + ".h5")
			generator.save("srgan_generator_epochs" + str(e + 1))
			save_images(valid_data, generator, e)

	# Randomly sample from validation data and perform super resolution
	# on that sample.
	random.seed(42)
	index = random.randint(0, len(list(valid_data.as_numpy_iterator())))
	sample = list(valid_data.as_numpy_iterator())[index]
	src_img = sample["lr"]
	tar_img = sample["hr"]

	loaded_generator = load_model(
		# "srgan_generator_epochs" + str(epochs) + ".h5",
		"srgan_generator_epochs" + str(epochs),
		custom_objects={
			"ResBlock": ResBlock, 
			"UpScaleBlock": UpScaleBlock,
		}
	)
	gen_img = loaded_generator.predict(src_img)

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
	plt.show()

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()