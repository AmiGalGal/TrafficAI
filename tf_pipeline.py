import tensorflow as tf
from preprocess import tf_preprocess, tf_preprocess_with_augmentation

TARGET_SIZE = 320

def build_dataset(img_paths, json_paths, batch_size=4, shuffle=True, augment=True):

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, json_paths))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(img_paths), reshuffle_each_iteration=True)

    preprocess_fn = tf_preprocess_with_augmentation if augment else tf_preprocess

    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset