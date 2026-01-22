import tensorflow as tf
from backbone import build_backbone
from detectionHead import build_detection_head
from segmentationHead import build_segmentation_head
from sceneHead import build_scene_head

def build_multitask_model(
    backbone_name="resnet50",
    train_backbone=False,
    num_detection_classes=10,
    num_segmentation_classes=11,
    num_scene_classes=5
):
    inputs, features, out_channels = build_backbone(
        backbone_name=backbone_name,
        trainable=train_backbone
    )

    detection_out = build_detection_head(features, num_detection_classes)
    segmentation_out = build_segmentation_head(features, num_segmentation_classes)
    scene_out = build_scene_head(features, num_scene_classes)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "detection": detection_out,
            "segmentation": segmentation_out,
            "scene": scene_out
        },
        name="multi_task_model"
    )

    return model
