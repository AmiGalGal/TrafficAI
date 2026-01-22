import tensorflow as tf

def build_scene_head(
    backbone_features,
    num_classes,
    name="scene_head"
):
    x = backbone_features

    x = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        name=f"{name}_dense1"
    )(x)

    scene_logits = tf.keras.layers.Dense(
        num_classes,
        activation=None,  # logits
        name=f"{name}_logits"
    )(x)

    return scene_logits