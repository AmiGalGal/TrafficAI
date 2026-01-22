import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class_map = {
    1: 'traffic light',
    2: 'traffic sign',
    3: 'car',
    4: 'person',
    5: 'bus',
    6: 'truck',
    7: 'rider',
    8: 'bike',
    9: 'motor',
    10: 'train'
}

def resize_boxes(boxes, orig_size, target_size):
    H_orig, W_orig = orig_size
    H_target, W_target = target_size
    scale_x = W_target / W_orig
    scale_y = H_target / H_orig

    boxes_resized = boxes.copy()
    boxes_resized[:, 0] *= scale_x  # x1
    boxes_resized[:, 2] *= scale_x  # x2
    boxes_resized[:, 1] *= scale_y  # y1
    boxes_resized[:, 3] *= scale_y  # y2
    return boxes_resized

def visualize_batch_from_dataset(dataset, seg_colors, box_colors, original_size=(720, 1280), max_samples=1):
    for batch_idx, (images, labels) in enumerate(dataset):
        batch_size = images.shape[0]
        seg_masks = labels['seg_mask']
        boxes_batch = labels['boxes']
        boxes_valid = labels['boxes_valid']
        scenes = labels['scene']

        for i in range(min(batch_size, max_samples)):
            img = images[i].numpy()
            mask = seg_masks[i].numpy()
            boxes = boxes_batch[i].numpy()
            valid = boxes_valid[i].numpy()
            scene_name = scenes[i].numpy().decode('utf-8') if isinstance(scenes[i].numpy(), bytes) else str(scenes[i].numpy())

            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
            for cat_id, color in seg_colors.items():
                mask_rgb[mask == cat_id] = np.array(color)/255.0  # Normalize for matplotlib

            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            ax[0].imshow(img)
            ax[0].set_title(f"Original Image\nScene: {scene_name}")
            ax[0].axis('off')

            ax[1].imshow(img)
            ax[1].imshow(mask_rgb, alpha=0.5)
            ax[1].set_title(f"Segmentation + Boxes\nScene: {scene_name}")
            ax[1].axis('off')

            if boxes is not None:
                valid_boxes = boxes[valid.astype(bool)]
                boxes_resized = resize_boxes(valid_boxes, original_size, (img.shape[0], img.shape[1]))
                for b in boxes_resized:
                    x1, y1, x2, y2, class_id = b
                    class_id = int(class_id)
                    color = box_colors.get(class_id, (255, 255, 255))  # default white
                    rect = Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2,
                        edgecolor=np.array(color)/255.0,
                        facecolor='none'
                    )
                    ax[1].add_patch(rect)

                    # Use class_map if available
                    label = class_map.get(class_id, str(class_id)) if class_map else str(class_id)
                    ax[1].text(
                        x1, y1-5,
                        label,
                        color=np.array(color)/255.0,
                        fontsize=9,
                        bbox=dict(facecolor='black', alpha=0.5, pad=1)
                    )

        plt.tight_layout()
        plt.show()
        break
