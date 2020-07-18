import numpy as np
import tensorflow as tf


def evaluate(model, data):
    psnr_values = []
    for lr, hr in data:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


def resolve_single(model, lr):
    return resolve_image(model, tf.expand_dims(lr, axis=0))[0]


def resolve_image(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


DIV2K_rgbmean = np.array([0.4488, 0.4371, 0.4040]) * 255


def normalize(x, rgb_mean=DIV2K_rgbmean):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_rgbmean):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    return x / 255.0


def normalize_m11(x):
    return x / 127.5 - 1


def denormalize_m11(x):
    return (x + 1) * 127.5

# Metrics


def pixel_shuffle(scaling):
    return lambda x: tf.nn.depth_to_space(x, scaling)


def psnr(X1, X2):
    return tf.image.psnr(X1, X2, max_val=255)
