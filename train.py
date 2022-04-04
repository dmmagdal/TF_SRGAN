# train.py
# Initialize and train a SRGAN model for image super resolution.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
from srgan import ResBlock, UpScaleBlock, DiscriminatorBlock
from srgan import create_generator, create_discriminator
from srgan import build_vgg19
from utils import resize_images, scale_images, save_images
from matplotlib import pyplot as plt


class SRGAN(keras.Model):
	def __init__(self, generator, discriminator, vgg, **kwargs):
		super(SRGAN, self).__init__(**kwargs)

		# Models.
		self.discriminator = discriminator
		self.generator = generator
		self.vgg = vgg

		# Mean squared error for VGG loss.
		self.mse = keras.losses.MeanSquaredError()


	def vgg_loss(self, y_true, y_pred):
		# Pass both the real and fake (generated) high res images
		# through the VGG19 model.
		real_features = self.vgg(y_true)
		fake_features = self.vgg(y_pred)

		# Return the MSE between the real and generated images.
		return self.mse(real_features, fake_features)


	def compile(self, gen_opt, disc_opt):
		super(SRGAN, self).compile()
		self.disc_optimizer = disc_opt
		self.gen_optimizer = gen_opt
		self.d_loss_metric = keras.metrics.Mean(name="d_loss")
		self.g_loss_metric = keras.metrics.Mean(name="g_loss")
		self.d_loss_fn = keras.losses.BinaryCrossentropy()
		self.g_loss_fn = self.vgg_loss # content loss
		self.g_loss_fn2 = keras.losses.MeanSquaredError()
		self.custom_loss_weights = [1e-3, 1]


	@property
	def metrics(self):
		return [self.d_loss_metric, self.g_loss_metric]


	def train_step(self, data):
		# Get the input (low resolution/lr and high resolution/hr
		# images). Also extract the batch size.
		lr_imgs = data["lr"]
		hr_imgs = data["hr"]
		batch_size = tf.shape(hr_imgs)[0]

		# Use the gradient tapes for both the discrinimator and
		# generator to track gradients for the respective models.
		with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
			# Generate fake images.
			fake_imgs = self.generator(lr_imgs)

			# Combine with real images.
			combined_imgs = tf.concat([fake_imgs, hr_imgs], axis=0)

			# Initialize labels for the respective images (1 for fake
			# images from the generator and 0 for real hr images).
			labels = tf.concat(
				[tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],
				axis=0
			)

			# Add some random noise to the labels (supposed to be an
			# important trick). See DCGAN example from Keras examples.
			#labels += 0.05 * tf.random.uniform(tf.shape(labels))

			# Train discriminator.
			predictions = self.discriminator(combined_imgs)
			d_loss = self.d_loss_fn(labels, predictions)
			# d_loss = self.d_loss_fn(labels, predictions) * self.custom_loss_weights[0]
			# d_loss = self.d_loss_fn(labels, predictions, self.custom_loss_weights[0])
			grads = disc_tape.gradient(
				d_loss, self.discriminator.trainable_weights
			)
			self.disc_optimizer.apply_gradients(
				zip(grads, self.discriminator.trainable_variables)
			)

			# Train generator (Do NOT update the weights of the
			# discriminator).
			g_loss = self.g_loss_fn(hr_imgs, fake_imgs) * self.custom_loss_weights[1]
			g_loss2 = self.d_loss_fn(
				#labels[:batch_size, :], predictions[:batch_size, :], self.custom_loss_weights[0]
				labels[-batch_size:, :], predictions[:batch_size, :], self.custom_loss_weights[0] # ***
				# ***) This line is to see how many Real [0] predictions made it past the discriminator.
				# We are trying to make sure that the generator fools the discriminator so we take the
				# loss of the predictions on the generated images with the number of times the discriminator
				# falsely predicted the image was "real"/0.
			)
			g_loss3 = self.g_loss_fn2(hr_imgs, fake_imgs, self.custom_loss_weights[1])
			grads = gen_tape.gradient(
				g_loss + g_loss2 + g_loss3, self.generator.trainable_weights
			)
			self.gen_optimizer.apply_gradients(
				zip(grads, self.generator.trainable_weights)
			)

		# Update metrics.
		self.d_loss_metric.update_state(d_loss)
		self.g_loss_metric.update_state(g_loss)
		return {
			"d_loss": self.d_loss_metric.result(),
			"g_loss": self.g_loss_metric.result(),
		}


	def save(self, path, h5=True):
		if h5:
			self.generator.save(path + "_generator.h5")
			self.discriminator.save(path + "_discriminator.h5")
		else:
			self.generator.save(path + "_generator")
			self.discriminator.save(path + "_discriminator")


class GANMonitor(keras.callbacks.Callback):
	def __init__(self, valid_data, epoch_freq=10):
		self.valid_data = valid_data
		self.epoch_freq = epoch_freq


	def on_epoch_end(self, epoch, logs=None):
		if (epoch + 1) % self.epoch_freq == 0:
			save_images(self.valid_data, self.model.generator, epoch, True)


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

	# Initialize models (generator, discriminator, and vgg).
	gen_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.99)
	generator = create_generator(lr_inputs, num_res_blocks=16)
	generator.summary()

	disc_opt = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.99)
	discriminator = create_discriminator(hr_inputs)
	discriminator.summary()

	vgg = build_vgg19(hr_shape)
	vgg.trainable = False
	vgg.summary()

	# Initialize SRGAN model.
	gan = SRGAN(generator, discriminator, vgg)
	gan.compile(gen_opt, disc_opt)
	save_callback = GANMonitor(valid_data)

	# Train the GAN.
	epochs = 500#1000#100
	batch_size = 4
	train_data = train_data.prefetch(buffer_size=autotune)\
		.batch(batch_size)
	valid_data = valid_data.prefetch(buffer_size=autotune)\
		.batch(batch_size)
	history = gan.fit(
		train_data,
		epochs=epochs,
		callbacks=[save_callback]
	)

	# Save the generator from the GAN.
	# gan.save(f"SRGAN_{epochs}", h5=True)
	gan.save(f"SRGAN_{epochs}", h5=False)

	# Randomly sample from validation data and perform super resolution
	# on that sample.
	random.seed(42)
	index = random.randint(0, len(list(valid_data.as_numpy_iterator())))
	sample = list(valid_data.as_numpy_iterator())[index]
	src_img = sample["lr"]
	tar_img = sample["hr"]

	loaded_generator = load_model(
		# "SRGAN_" + str(epochs) + "_generator.h5",
		"SRGAN_" + str(epochs) + "_generator",
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