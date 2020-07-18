from utils import load_image
from model import resolve_single
import os
import matplotlib.pyplot as plt

from data import DIV2K
from model.srgan import generator, discriminator
from train import SrganTrainer, SrganGeneratorTrainer
import tensorflow as tf
# print(tf.VERSION)

weights_dir = 'weights'


def weights_file(filename): return os.path.join(weights_dir, filename)


os.makedirs(weights_dir, exist_ok=True)

pre_generator = generator()
gan_generator = generator()

pre_generator.load_weights(weights_file('pre_generator.h5'))
gan_generator.load_weights(weights_file('gan_generator.h5'))


def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)
    model = gan_generator
    gan_sr = resolve_single(model, lr)
    plt.figure(figsize=(70, 70))

    images = [lr, gan_sr]
    titles = ['LR Input', 'SR Output']
    positions = [1, 2]

    for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
        plt.subplot(1, 2, pos, autoscale_on=True)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

    plt.show()


# Cat
# resolve_and_plot('demo/image1-91x121.png')

# Kenny
# resolve_and_plot('demo/image2-165x165.jpg')
