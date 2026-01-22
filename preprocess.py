import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data import load_image, load_json, extract_segmentation_polygons, extract_detection_objects
from matplotlib.patches import Rectangle
from augmentations import apply_color_augmentation


TARGET_SIZE = 320
MAX_BOXES = 150

SEGMENTATION_LABELS = {
    'area/drivable': 1,
    'lane/road curb': 2,
    'area/alternative': 3,
    'lane/single white': 4,
    'lane/double yellow': 5,
    'lane/single yellow': 6,
    'lane/crosswalk': 7,
    'lane/double white': 8,
    'lane/single other': 9,
    'lane/double other': 10,
    'area/unknown': 11
}

LABEL_COLORS = {
    1: (255, 0, 0),       # red
    2: (255, 255, 0),     # yellow
    3: (0, 0, 255),       # blue
    4: (0, 255, 0),       # green
    5: (255, 128, 0),     # orange
    6: (128, 255, 0),     # lime
    7: (255, 0, 128),     # pink
    8: (0, 255, 128),     # teal
    9: (128, 0, 255),     # purple
    10: (0, 128, 255),    # sky blue
    11: (128, 128, 128)   # gray
}

CLASS_MAP = {
    'traffic light': 1,
    'traffic sign': 2,
    'car': 3,
    'person': 4,
    'bus': 5,
    'truck': 6,
    'rider': 7,
    'bike': 8,
    'motor': 9,
    'train': 10
}

OBJECT_COLORS = {
    1: (255, 0, 0),  # red
    2: (255, 255, 0),  # yellow
    3: (0, 0, 255),  # blue
    4: (0, 255, 0),  # green
    5: (255, 128, 0),  # orange
    6: (128, 255, 0),  # lime
    7: (255, 0, 128),  # pink
    8: (0, 255, 128),  # teal
    9: (128, 0, 255),  # purple
    10: (0, 128, 255),  # sky blue
}

def resize_img(img, target_size = TARGET_SIZE):
    img = cv2.resize(img, (target_size, target_size))
    return img

def normalize(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img

def build_segmentation_mask(poly2d_objects, mask_h, mask_w, label_map, orig_h, orig_w):
    mask = np.zeros((mask_h, mask_w), dtype=np.int32)
    valid = False

    for obj in poly2d_objects:
        category = obj.get("category", "area/unknown")
        if category not in label_map:
            continue
        cat_id = label_map[category]

        poly2d_list = obj["poly2d"]

        # Scale points to mask size
        points = []
        for pt in poly2d_list:
            if isinstance(pt, (list, tuple)) and len(pt) == 3:
                x, y, t = pt
                x_scaled = int(x / orig_w * mask_w)
                y_scaled = int(y / orig_h * mask_h)
                points.append((x_scaled, y_scaled))

        if len(points) < 2:
            continue

        pts = np.array(points, dtype=np.int32)

        if len(points) == 2:
            cv2.line(mask, tuple(pts[0]), tuple(pts[1]), color=cat_id, thickness=4)
        else:
            cv2.fillPoly(mask, [pts], color=cat_id)

        valid = True

    return mask, valid

def visualize_image_and_mask(img, mask, label_colors, boxes = None, orig_size=None):
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)

    for cat_id, color in label_colors.items():
        rgb = color[:3]  # take only RGB
        mask_rgb[mask == cat_id] = rgb

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Overlay
    axes[1].imshow(img)
    axes[1].imshow(mask_rgb, alpha=0.5)  # alpha from color dictionary
    axes[1].set_title("Segmentation Overlay")
    axes[1].axis('off')

    if boxes is not None and orig_size is not None:
        boxes_resized = resize_boxes(boxes, orig_size, (img.shape[0], img.shape[1]))
        for b in boxes_resized:
            x1, y1, x2, y2 = b['bbox']
            label = b['label']
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
            axes[1].add_patch(rect)
            axes[1].text(x1, y1-5, label, color='lime', fontsize=9,
                         bbox=dict(facecolor='black', alpha=0.5, pad=1))

    plt.tight_layout()
    plt.show()

def resize_boxes(boxes, orig_size, TARGET_SIZE):
    H_orig, W_orig = orig_size
    H_new, W_new = TARGET_SIZE
    scale_x = W_new / W_orig
    scale_y = H_new / H_orig

    resized_boxes = []
    for b in boxes:
        x1, y1, x2, y2 = b['bbox']
        resized_boxes.append({
            'label': b['label'],
            'bbox': (
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            )
        })
    return resized_boxes

def testing(img_path, json_path):

    ann = load_json(json_path)
    frame = ann["frames"][0]
    poly_objs = extract_segmentation_polygons(frame)
    img = load_image(img_path)
    original_size = (720, 1280)

    img_resized = resize_img(img)
    img_resized = normalize(img_resized)

    boxes = extract_detection_objects(frame)
    mask, valid = build_segmentation_mask(poly_objs, TARGET_SIZE, TARGET_SIZE, SEGMENTATION_LABELS, original_size[0], original_size[1])

    print("Mask shape:", mask.shape)
    print("Valid segmentation?", valid)
    print("Unique IDs in mask:", np.unique(mask))

    visualize_image_and_mask(img_resized, mask, LABEL_COLORS, boxes, original_size)

def boxes_to_tensor(boxes):
    box_tensor = np.zeros((MAX_BOXES, 5), dtype=np.float32)
    valid_mask = np.zeros((MAX_BOXES,), dtype=np.bool_)

    for i, b in enumerate(boxes[:MAX_BOXES]):
        box_tensor[i, :4] = b['bbox']  # x1,y1,x2,y2
        box_tensor[i, 4] = CLASS_MAP.get(b['label'], 0)  # class id
        valid_mask[i] = True

    return box_tensor, valid_mask

def preprocess_sample(img_path, json_path):
    # Convert tf.Tensor strings to Python strings
    if isinstance(img_path, tf.Tensor):
        img_path = img_path.numpy().decode('utf-8')
    if isinstance(json_path, tf.Tensor):
        json_path = json_path.numpy().decode('utf-8')

    img = load_image(img_path)
    ann = load_json(json_path)
    frame = ann["frames"][0]
    boxes = extract_detection_objects(frame)
    poly_objs = extract_segmentation_polygons(frame)
    scene = ann["attributes"]["scene"]

    original_size = (720, 1280)

    img = resize_img(img)  # TARGET_SIZE x TARGET_SIZE
    img = normalize(img)

    mask, valid_mask = build_segmentation_mask(
        poly_objs, TARGET_SIZE, TARGET_SIZE, SEGMENTATION_LABELS, original_size[0], original_size[1]
    )

    # Convert to tensors using tf.convert_to_tensor / tf.cast
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    mask_tensor = tf.convert_to_tensor(mask, dtype=tf.int32)
    boxes_tensor, boxes_valid = boxes_to_tensor(boxes)  # returns tensors
    valid_tensor = tf.convert_to_tensor(valid_mask, dtype=tf.bool)

    scene_tensor = tf.convert_to_tensor(scene, dtype=tf.string)

    return img_tensor, scene_tensor, boxes_tensor, boxes_valid, mask_tensor, valid_tensor

def tf_preprocess(img_path, json_path):
    img, scene, boxes, boxes_valid, mask, seg_valid = tf.py_function(
        func=preprocess_sample,
        inp=[img_path, json_path],
        Tout=[tf.float32, tf.string, tf.float32, tf.bool, tf.int32, tf.bool]
    )

    img.set_shape([TARGET_SIZE, TARGET_SIZE, 3])
    mask.set_shape([TARGET_SIZE, TARGET_SIZE])
    boxes.set_shape([MAX_BOXES, 5])
    boxes_valid.set_shape([MAX_BOXES])

    return img, {"scene": scene, "boxes": boxes, "boxes_valid": boxes_valid, "seg_mask": mask, "seg_valid": seg_valid}

def tf_preprocess_with_augmentation(img_path, json_path):
    img, labels = tf_preprocess(img_path, json_path)

    img = apply_color_augmentation(img)

    return img, labels


IMG_HEIGHT, IMG_WIDTH = 512, 512
def preprocess_image(pil_image):
    img = np.array(pil_image).astype(np.float32)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    return img