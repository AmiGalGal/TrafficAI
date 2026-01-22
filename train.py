import os
import tensorflow as tf
import datetime

from metrics import (
    detection_mean_iou,
    segmentation_pixel_accuracy,
    segmentation_mean_iou,
    scene_accuracy
)
from tf_pipeline import build_dataset
from multiTask import build_multitask_model

# =========================
# CONFIG
# =========================

LOG_DIR = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(LOG_DIR)
count = 0

IMG_DIR_TRAIN = "images/train"
IMG_DIR_VAL   = "images/val"
LBL_DIR_TRAIN = "labels/train"
LBL_DIR_VAL   = "labels/val"

BATCH_SIZE = 4
EPOCHS_HEADS = 5
EPOCHS_FULL = 15
LEARNING_RATE = 1e-4

MAX_BOXES = 150
NUM_DET_CLASSES = 11
NUM_SEG_CLASSES = 12
NUM_SCENE_CLASSES = 7

SCENE_CLASSES = [
    'city street', 'highway', 'residential',
    'parking lot', 'undefined', 'tunnel', 'gas stations'
]

SCENE_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(SCENE_CLASSES),
        values=tf.constant(list(range(len(SCENE_CLASSES))), dtype=tf.int64)
    ),
    default_value=4  # undefined
)

def list_files(img_dir, lbl_dir):
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")])
    labels = [os.path.join(lbl_dir, os.path.basename(f).replace(".jpg", ".json")) for f in images]
    return images, labels

def detection_loss(y_true, y_pred):
    true_boxes = y_true["boxes"][..., :4]
    true_classes = tf.cast(y_true["boxes"][..., 4], tf.int32)
    valid = tf.cast(y_true["boxes_valid"], tf.float32)

    pred_boxes = y_pred["boxes"]
    pred_logits = y_pred["class_logits"]
    pred_obj = tf.squeeze(y_pred["objectness"], -1)
    pred_obj = tf.clip_by_value(pred_obj, 1e-6, 1.0 - 1e-6)

    delta = 1.0
    diff = true_boxes - pred_boxes
    abs_diff = tf.abs(diff)
    huber_loss = tf.where(abs_diff < delta, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta))
    box_loss = tf.reduce_sum(huber_loss, axis=-1)  # [B,150]
    box_loss = box_loss * valid
    box_loss = tf.reduce_sum(box_loss) / (tf.reduce_sum(valid) + 1e-6)

    cls_mask = tf.logical_and(valid > 0, true_classes > 0)
    cls_mask = tf.cast(cls_mask, tf.float32)
    cls_targets = tf.maximum(true_classes - 1, 0)  # 0-based for sparse_softmax
    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=cls_targets,
        logits=pred_logits
    )
    cls_loss = tf.reduce_sum(cls_loss * cls_mask) / (tf.reduce_sum(cls_mask) + 1e-6)

    obj_loss = tf.keras.losses.binary_crossentropy(valid, pred_obj)
    obj_loss = tf.reduce_mean(obj_loss)

    return 0.0001*(box_loss + cls_loss + obj_loss)

def segmentation_loss(y_true, y_pred):
    mask = y_true["seg_mask"]
    valid = tf.cast(y_true["seg_valid"], tf.float32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=mask,
        logits=y_pred
    )
    loss = tf.reduce_mean(loss, axis=[1,2])
    return tf.reduce_mean(loss * valid)

def scene_loss(y_true, y_pred):
    scene_ids = SCENE_TABLE.lookup(y_true)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=scene_ids,
        logits=y_pred
    )
    return tf.reduce_mean(loss)


@tf.function
def train_step(model, optimizer, images, labels):
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)

        d_loss = detection_loss(labels, outputs["detection"])
        s_loss = segmentation_loss(labels, outputs["segmentation"])
        c_loss = scene_loss(labels["scene"], outputs["scene"])

        total_loss = d_loss + s_loss + c_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return d_loss, s_loss, c_loss, total_loss


def train(model, dataset, optimizer, epochs, val_ds):
    best_val = float("inf")
    patience = 3
    bad_epochs = 0
    global count
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        for step, (images, labels) in enumerate(dataset):
            d, s, c, t = train_step(model, optimizer, images, labels)
            count += 1
            if step % 20 == 0:
                print(
                    f"step {step:04d} | "
                    f"det {d:.3f} | seg {s:.3f} | scene {c:.3f} | total {t:.3f}"
                )
            with writer.as_default():
                tf.summary.scalar("train/detection_loss", d, step=count)
                tf.summary.scalar("train/segmentation_loss", s, step=count)
                tf.summary.scalar("train/scene_loss", c, step=count)
                tf.summary.scalar("train/total_loss", t, step=count)
        # Validation loop
        val_det, val_seg, val_scene, val_total = 0, 0, 0, 0
        val_iou, val_seg_acc, val_seg_iou, val_scene_acc = 0, 0, 0, 0
        val_steps = 0
        for val_images, val_labels in val_ds:
            d_loss, s_loss, c_loss, t_loss, det_iou, seg_acc, seg_iou, scn_acc = validate_step(model, val_images, val_labels)
            val_det += d_loss
            val_seg += s_loss
            val_scene += c_loss
            val_total += t_loss

            val_iou += det_iou
            val_seg_acc += seg_acc
            val_seg_iou += seg_iou
            val_scene_acc += scn_acc

            val_steps += 1

        print(
            f"VAL | "
            f"det {val_det / val_steps:.3f} | "
            f"seg {val_seg / val_steps:.3f} | "
            f"scene {val_scene / val_steps:.3f} | "
            f"total {val_total / val_steps:.3f} | "
            f"IoU {val_iou / val_steps:.3f} | "
            f"segAcc {val_seg_acc / val_steps:.3f} | "
            f"segIoU {val_seg_iou / val_steps:.3f} | "
            f"sceneAcc {val_scene_acc / val_steps:.3f}"
            )
        with writer.as_default():
            tf.summary.scalar("val/detection_loss", val_det / val_steps, step=epoch)
            tf.summary.scalar("val/segmentation_loss", val_seg / val_steps, step=epoch)
            tf.summary.scalar("val/scene_loss", val_scene / val_steps, step=epoch)
            tf.summary.scalar("val/total_loss", val_total / val_steps, step=epoch)

            tf.summary.scalar("val/detection_iou", val_iou / val_steps, step=epoch)
            tf.summary.scalar("val/segmentation_iou", val_seg_iou / val_steps, step=epoch)
            tf.summary.scalar("val/scene_accuracy", val_scene_acc / val_steps, step=epoch)
        if val_total / val_steps < best_val:
            best_val = val_total / val_steps
            bad_epochs = 0
            model.save("best_model.keras")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("🛑 Early stopping triggered")
            return

def validate_step(model, images, labels):
    outputs = model(images, training=False)

    det_iou = detection_mean_iou(labels, outputs["detection"])
    seg_acc = segmentation_pixel_accuracy(labels, outputs["segmentation"])
    seg_iou = segmentation_mean_iou(labels, outputs["segmentation"], NUM_SEG_CLASSES)
    scn_acc = scene_accuracy(labels["scene"], outputs["scene"], SCENE_TABLE)

    d_loss = detection_loss(labels, outputs["detection"])
    s_loss = segmentation_loss(labels, outputs["segmentation"])
    c_loss = scene_loss(labels["scene"], outputs["scene"])

    total_loss = d_loss + s_loss + c_loss

    return d_loss, s_loss, c_loss, total_loss, det_iou, seg_acc, seg_iou, scn_acc



lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=20000,
    alpha=1e-6
)

def main():
    # Files
    train_imgs, train_lbls = list_files(IMG_DIR_TRAIN, LBL_DIR_TRAIN)
    val_imgs, val_lbls = list_files(IMG_DIR_VAL, LBL_DIR_VAL)

    # Datasets
    train_ds = build_dataset(
        train_imgs, train_lbls,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True
    )

    val_ds = build_dataset(
        val_imgs, val_lbls,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )

    # Model
    model = build_multitask_model(
        backbone_name="resnet50",
        train_backbone=False,
        num_detection_classes=NUM_DET_CLASSES,
        num_segmentation_classes=NUM_SEG_CLASSES,
        num_scene_classes=NUM_SCENE_CLASSES
    )

    model.summary()

    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    print("\nTraining heads only")
    for layer in model.layers:
        if "resnet" in layer.name:
            layer.trainable = False

    train(model, train_ds, optimizer, EPOCHS_HEADS, val_ds)

    print("\nFine-tuning full model")
    for layer in model.layers:
        layer.trainable = True

    train(model, train_ds, optimizer, EPOCHS_FULL, val_ds)

if __name__ == "__main__":
    main()
