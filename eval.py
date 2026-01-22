from metrics import detection_mean_iou, segmentation_pixel_accuracy, segmentation_mean_iou, scene_accuracy
import tensorflow as tf
from tf_pipeline import build_dataset
import os

NUM_SEG_CLASSES = 12
MODEL_PATH = "best_model.keras"
BATCH_SIZE = 4
imgs_path = "images/test"
labels_path = "labels/test"

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

def eval_step(model, images, labels):
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

def evaluation(model, eval_ds):
    eval_det, eval_seg, eval_scene, eval_total = 0, 0, 0, 0
    eval_iou, eval_seg_acc, eval_seg_iou, eval_scene_acc = 0, 0, 0, 0
    val_steps = 0
    for val_images, val_labels in eval_ds:
        d_loss, s_loss, c_loss, t_loss, det_iou, seg_acc, seg_iou, scn_acc = eval_step(model, val_images,
                                                                                           val_labels)
        eval_det += d_loss
        eval_seg += s_loss
        eval_scene += c_loss
        eval_total += t_loss

        eval_iou += det_iou
        eval_seg_acc += seg_acc
        eval_seg_iou += seg_iou
        eval_scene_acc += scn_acc

        val_steps += 1

    print(
        f"VAL | "
        f"det {eval_det / val_steps:.3f} | "
        f"seg {eval_seg / val_steps:.3f} | "
        f"scene {eval_scene / val_steps:.3f} | "
        f"total {eval_total / val_steps:.3f} | "
        f"IoU {eval_iou / val_steps:.3f} | "
        f"segAcc {eval_seg_acc / val_steps:.3f} | "
        f"segIoU {eval_seg_iou / val_steps:.3f} | "
        f"sceneAcc {eval_scene_acc / val_steps:.3f}"
    )

def main():
    eval_imgs, eval_lbls = list_files(imgs_path, labels_path)
    eval_ds = build_dataset(
        eval_imgs, eval_lbls,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    evaluation(model, eval_ds)


main()
