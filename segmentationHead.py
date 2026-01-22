import tensorflow as tf

def build_segmentation_head(features, num_classes, name="segmentation"):
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name=f"{name}_conv1")(features)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name=f"{name}_conv2")(x)
    x = tf.keras.layers.Conv2D(num_classes, 1, padding='same', name=f"{name}_logits")(x)

    x = tf.keras.layers.Resizing(
        height=320,
        width=320,
        interpolation='bilinear',
        name=f"{name}_resize"
    )(x)

    return x
