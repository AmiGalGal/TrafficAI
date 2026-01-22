import tensorflow as tf

def build_backbone(
    inputs=None,
    backbone_name="resnet50",
    trainable=False
):

    if inputs is None:
        # create input if not provided
        inputs = tf.keras.Input(shape=(320, 320, 3), name="image")

    if backbone_name == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
        output_channels = 2048

    elif backbone_name == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
        output_channels = 1280

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    base_model.trainable = trainable

    features = base_model.output

    return inputs, features, output_channels
