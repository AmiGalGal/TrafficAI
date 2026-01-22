import tensorflow as tf

def build_detection_head(
    backbone_features,
    num_classes=10,
    max_boxes=150,
    hidden_dim=256,
    name="detection_head"
):

    x = backbone_features

    x = tf.keras.layers.Conv2D(
        hidden_dim,
        kernel_size=3,
        padding="same",
        activation="relu",
        name=f"{name}_conv1"
    )(x)

    x = tf.keras.layers.Conv2D(
        hidden_dim,
        kernel_size=3,
        padding="same",
        activation="relu",
        name=f"{name}_conv2"
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name=f"{name}_gap"
    )(x)

    x = tf.keras.layers.Dense(
        hidden_dim,
        activation="relu",
        name=f"{name}_fc"
    )(x)

    box_output = tf.keras.layers.Dense(
        max_boxes * 4,
        name=f"{name}_boxes"
    )(x)

    box_output = tf.keras.layers.Reshape(
        (max_boxes, 4),
        name=f"{name}_boxes_reshape"
    )(box_output)

    class_logits = tf.keras.layers.Dense(
        max_boxes * num_classes,
        name=f"{name}_class_logits"
    )(x)

    class_logits = tf.keras.layers.Reshape(
        (max_boxes, num_classes),
        name=f"{name}_class_reshape"
    )(class_logits)

    objectness = tf.keras.layers.Dense(
        max_boxes,
        activation="sigmoid",
        name=f"{name}_objectness"
    )(x)

    objectness = tf.keras.layers.Reshape(
        (max_boxes, 1),
        name=f"{name}_objectness_reshape"
    )(objectness)

    return {
        "boxes": box_output,
        "class_logits": class_logits,
        "objectness": objectness
    }

#not a YOLO model, YOLO best work alone and in addition to the other performance might take a hit