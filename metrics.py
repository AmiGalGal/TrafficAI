import tensorflow as tf

def box_iou(boxes1, boxes2):

    x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])

    inter_w = tf.maximum(0.0, x2 - x1)
    inter_h = tf.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union = area1 + area2 - inter_area + 1e-6
    return inter_area / union


def detection_mean_iou(y_true, y_pred, iou_threshold=0.5):
    true_boxes = y_true["boxes"][..., :4]
    pred_boxes = y_pred["boxes"]
    valid = tf.cast(y_true["boxes_valid"], tf.bool)

    # Flatten valid boxes
    true_boxes = tf.boolean_mask(true_boxes, valid)
    pred_boxes = tf.boolean_mask(pred_boxes, valid)

    if tf.shape(true_boxes)[0] == 0:
        return tf.constant(0.0)

    ious = box_iou(true_boxes, pred_boxes)
    return tf.reduce_mean(ious)


def segmentation_pixel_accuracy(y_true, y_pred):

    gt = y_true["seg_mask"]
    valid = tf.cast(y_true["seg_valid"], tf.float32)

    pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    correct = tf.cast(tf.equal(gt, pred), tf.float32)
    correct = tf.reduce_mean(correct, axis=[1,2])

    return tf.reduce_mean(correct * valid)


def segmentation_mean_iou(y_true, y_pred, num_classes):
    gt = y_true["seg_mask"]
    pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    ious = []

    for cls in range(num_classes):
        gt_c = tf.equal(gt, cls)
        pred_c = tf.equal(pred, cls)

        intersection = tf.reduce_sum(tf.cast(gt_c & pred_c, tf.float32))
        union = tf.reduce_sum(tf.cast(gt_c | pred_c, tf.float32))

        iou = tf.where(union > 0, intersection / (union + 1e-6), 0.0)
        ious.append(iou)

    return tf.reduce_mean(tf.stack(ious))

def scene_accuracy(y_true_scene, y_pred_scene, scene_table):
    true_ids = scene_table.lookup(y_true_scene)
    preds = tf.argmax(y_pred_scene, axis=-1, output_type=tf.int64)
    acc = tf.reduce_mean(tf.cast(tf.equal(true_ids, preds), tf.float32))
    return acc
