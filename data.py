import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import os
from matplotlib.lines import Line2D

def load_image(path):
    img = cv2.imread(path)
    assert img is not None, path
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_detection_objects(frame):
    boxes = []

    for obj in frame["objects"]:
        if "box2d" not in obj:
            continue

        label = obj["category"]
        box = obj["box2d"]

        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        boxes.append({
            "label": label,
            "bbox": (x1, y1, x2, y2)
        })

    return boxes

def extract_segmentation_polygons(frame):
    polygons = []

    for obj in frame["objects"]:
        if "poly2d" not in obj:
            continue
        if len(obj["poly2d"]) == 0:
            continue

        polygons.append({
            "category": obj.get("category", "unknown"),
            "poly2d": obj["poly2d"]
        })

    return polygons

def poly2d_to_mpl_patches(obj):
    category = obj.get("category", "unknown")
    color_map = {
        "area/drivable": (1, 0, 0, 0.3),
        "lane/road curb": (1, 1, 0, 0.3),
        "area/alternative": (0, 0, 1, 0.3),
        "lane/single white": (0, 1, 0, 0.3),
        "lane/double yellow": (1, 0.5, 0, 0.3),
        "lane/single yellow": (0.5, 1, 0, 0.3),
        "lane/crosswalk": (1, 0, 0.5, 0.3),
        "lane/double white": (0, 1, 0.5, 0.3),
        "lane/single other": (0.5, 0, 1, 0.3),
        "lane/double other": (0, 0.5, 1, 0.3),
        "area/unknown": (0.5, 0.5, 0.5, 0.3)
    }
    facecolor = color_map.get(category, (1, 0, 1, 0.3))
    edgecolor = facecolor

    points = [(x, y) for x, y, t in obj["poly2d"] if t in ("M", "L", "C")]

    if len(points) < 2:
        return None  # nothing to draw

    if len(points) == 2:
        # Draw a line for 2-point polygons
        return Line2D([points[0][0], points[1][0]], [points[0][1], points[1][1]],
                      color=edgecolor, linewidth=2)
    else:
        # Draw filled polygon for 3+ points
        return Polygon(points, closed=True, facecolor=facecolor, edgecolor="none")

def testing():
    name = "fe189115-11bedd21"
    json_data = "labels/train/" + name + ".json"
    image = "images/train/" + name + ".jpg"

    ann = load_json(json_data)
    frame = ann["frames"][0]

    img = load_image(image)
    scene = ann["attributes"]["scene"]
    boxes = extract_detection_objects(frame)
    road_poly2d = extract_segmentation_polygons(frame)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.set_title(f"Scene: {scene}", fontsize=14)
    ax.axis('off')
    print(road_poly2d)
    for obj in road_poly2d:
        patch = poly2d_to_mpl_patches(obj)
        if patch is not None:
            if isinstance(patch, Polygon):
                ax.add_patch(patch)
            else:
                # It's a Line2D
                ax.add_line(patch)

    for b in boxes:
        x1, y1, x2, y2 = b['bbox']
        label = b['label']

        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        ax.text(x1, y1 - 5, label, color='lime', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.5, pad=1))

    plt.tight_layout()
    plt.show()

def checks():
    directories = ["labels/train", "labels/val", "labels/test"]
    missing_scene = []
    missing_detection = []
    missing_segmentation = []

    scenes = []
    detection_classes = []
    segmentation_classes = []

    for dir in directories:
        for fname in os.listdir(dir):
            if not fname.endswith(".json"):
                continue

            path = os.path.join(dir, fname)
            ann = load_json(path)

            # check scene
            if "scene" not in ann.get("attributes", {}) or not ann["attributes"]["scene"]:
                missing_scene.append(fname)
            else:
                if ann["attributes"]["scene"] not in scenes:
                    scenes.append(ann["attributes"]["scene"])

            frames = ann.get("frames", [])
            if len(frames) == 0:
                missing_detection.append(fname)
                missing_segmentation.append(fname)
                continue

            frame = frames[0]

            # detection
            boxes = extract_detection_objects(frame)
            if len(boxes) == 0:
                missing_detection.append(fname)
            for i in boxes:
                if i["label"] not in detection_classes:
                    detection_classes.append(i["label"])

            # segmentation
            polys = extract_segmentation_polygons(frame)
            if len(polys) == 0:
                missing_segmentation.append(fname)
            for i in polys:
                if i["category"] not in segmentation_classes:
                    segmentation_classes.append(i["category"])

    print("Images missing scene:", len(missing_scene))
    print("Images missing detection:", len(missing_detection))
    print("Images missing segmentation or no segments:", len(missing_segmentation))
    print(missing_segmentation)

    print("scene classes: ", scenes)
    print("detection classes: ", detection_classes)
    print("segmentation classes: ", segmentation_classes)
