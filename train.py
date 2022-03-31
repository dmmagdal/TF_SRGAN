# train.py
# Initialize and train a SRGAN model for image super resolution.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
from srgan import create_generator, create_discriminator
from srgan import create_combined_model
from utils import VGGLoss, resize_images, scale_images


class SRGAN(keras.Model):
	def __init__(self, generator, discriminator, gen_loss, disc_loss, gen_opt, disc_opt, **kwargs):
		# Generator/Discriminator models.
		self.gen = generator
		self.disc = discriminator

		# Generator/Discriminator losses.
		self.gen_loss = gen_loss
		self.disc_loss = disc_loss

		# Generator/Discriminator optimizers.
		self.gen_optimizer = gen_opt
		self.disc_optimizer = disc_opt


	def train_step(self, data):
		lr_imgs = data["lr"]
		hr_imgs = data["hr"]
		batch_size = tf.shape(hr_imgs)[0]

		fake_label = np.zeros((batch_size, 1))
		real_label = np.ones((batch_size, 1))

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			fake_imgs = self.gen(lr_imgs)

			self.disc.trainable = True
			real_outputs = self.disc(hr_imgs)
			fake_outputs = self.disc(fake_imgs)
			self.disc.trainable = False

			gen_loss = self.gen_loss(hr_imgs, fake_imgs)
			disc_loss = self.disc_loss(real_outputs, fake_outputs)

		gen_gradients = gen_tape.gradient(
			gen_loss, self.gen.trainable_variables
		)
		disc_gradients = disc_tape.gradient(
			disc_loss, self.disc.trainable_variables
		)
		
		self.gen_optimizer.apply_gradients(
			zip(gen_gradients, self.gen.trainable_variables)
		)
		self.disc_optimizer.apply_gradients(
			zip(disc_gradients, self.disc.trainable_variables)
		)


	def save(self, path):
		self.generator.save(path)


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
	vgg_loss = VGGLoss(hr_inputs)
	vgg_opt = keras.optimizers.Adam()
	generator = create_generator(lr_inputs, num_res_blocks=16)
	# generator.compile(loss=vgg_loss, optimizer="adam")
	generator.summary()

	disc_loss = keras.losses.BinaryCrossentropy(from_logits=True)
	disc_opt = keras.optimizers.Adam()
	discriminator = create_discriminator(hr_inputs)
	# discriminator.compile(
	# 	loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
	# )
	discriminator.summary()

	
	gan = SRGAN(
		generator, discriminator, vgg_loss, disc_loss,
		gen_opt, disc_opt
	)

	epochs = 100
	batch_size = 4
	train_data = train_data.prefetch(buffer_size=autotune).batch(batch_size)
	valid_data = valid_data.prefetch(buffer_size=autotune).batch(batch_size)
	history = model.fit(
		train_data,
		epochs=epochs,
		validation_data=valid_data 
	)
	model.save(f"generator_{epochs}.h5")


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()