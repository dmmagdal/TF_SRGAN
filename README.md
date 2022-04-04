# SRGAN

Description: This repository takes from multiple examples of the SRGAN for image super resolution and implements the model in Tensorflow 2.

How to Use:

 > Install the required modules from requirements.txt with `pip install -r requirements.txt`. The best way to train SRGAN from scratch is to use the training loop defined in `train.py`. Simply run `python train.py` and it will download the dataset (div2k/bicubic_x4 from tensorflow datasets) and begin training the neural network. You can go inside and alter the training hyperparameters (ie `batch_size`, `epochs`, etc), making this repo very easy to use for training the model from scratch.


### Sources:

 - [YouTube] (https://www.youtube.com/watch?v=1HqjPqNglPc&t=472s&ab_channel=DigitalSreeni)
 - [Github] (https://github.com/bnsreenu/python_for_microscopists/blob/master/255_256_SRGAN/SRGAN_train.py)
 - [YouTube Dataset] (https://press.liacs.nl/mirflickr/mirdownload.html)
 - [Original SRGAN Paper] (https://arxiv.org/pdf/1609.04802.pdf)
 - [Supplimentary YouTube Video] (https://www.youtube.com/watch?v=FwvTsx_dxn8&ab_channel=AIExpedition)
 - [Reference Blog] (https://blog.paperspace.com/super-resolution-generative-adversarial-networks/)
 - [DGGAN Keras Example] (https://keras.io/examples/generative/dcgan_overriding_train_step/)


### Training Photos:

1) A 500 epoch run with the training loop in train.py. This training loop removed the noise from the labels because labels were reaching beyond the required range of \[0, 1\], causing errors and issues with the discriminator loss (BCE).

2) A 500 epoch run with the training loop in train.py. This training loop included a weighted loss to the discriminator and the generator loss function included the Vgg-MSE (aka content-loss) and weighted BCE from discriminator (perceptual loss). 

3) A 500 epoch run with the training loop in train.py. This training loop included the following loss functions for the generator: Vgg-MSE (aka content loss) and weighted BCE from discriminator (perceptual loss). Note that there is still some distortion or "glitching" in the images despite being very similar to the high resolution image. This run also took care of the issues with matplotlib, in which we now no longer multiplied outputs from the generator by 255 (to convert to RGB), as matplotlib is already able to convert from grayscale to color on its own.

4) A 500 epoch run with the training loop in train.py. This training loop included the following loss functions for the generator: Vgg-MSE (aka content loss), weighted BCE from discriminator (perceptual loss), and raw MSE between the high resolution and generated images. This iteration, while capturing more details, was considered not as good because the outcome looked smudged compared to the previous run on the list.

5) A 100 epoch run with the training loop in srgan.py. Not a very good output.