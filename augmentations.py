import tensorflow as tf

def random_brightness(image, max_delta=0.2):
    return tf.image.random_brightness(image, max_delta)

def random_contrast(image, lower=0.8, upper=1.2):
    return tf.image.random_contrast(image, lower, upper)

def random_saturation(image, lower=0.8, upper=1.2):
    return tf.image.random_saturation(image, lower, upper)

def random_hue(image, max_delta=0.05):
    return tf.image.random_hue(image, max_delta)

def random_blur(image, kernel_size=3):
    image = tf.expand_dims(image, 0)  # add batch dim
    image = tf.nn.avg_pool(image, ksize=[1, kernel_size, kernel_size, 1], strides=[1,1,1,1], padding='SAME')
    return tf.squeeze(image, 0)

def apply_color_augmentation(image):
    image = random_brightness(image)
    image = random_contrast(image)
    image = random_saturation(image)
    image = random_hue(image)
    if tf.random.uniform([]) < 0.3:
        image = random_blur(image)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


