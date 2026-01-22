import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

# =========================
# CONFIG
# =========================

MODEL_PATH = "best_model.keras"

CLASS_MAP = {
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

SEG_COLORS = {
    0: (0, 0, 0),
    1: (128, 64,128),
    2: (244, 35,232),
    3: (70, 70, 70),
    4: (102,102,156),
    5: (190,153,153),
}

SCENE_CLASSES = [
    'city street', 'highway', 'residential',
    'parking lot', 'undefined', 'tunnel', 'gas stations'
]

OBJ_THRESH = 0.5

# =========================
# LOAD MODEL
# =========================

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model_input_shape = model.input_shape[1:3]
print("Model input:", model_input_shape)

# =========================
# INFERENCE
# =========================

def run_inference(image_pil):
    orig_w, orig_h = image_pil.size

    img = image_pil.resize(model_input_shape)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np[None, ...]

    outputs = model(img_np, training=False)

    # -------- Segmentation --------
    seg_logits = outputs["segmentation"][0]
    seg_mask = tf.argmax(seg_logits, axis=-1).numpy()

    # -------- Scene --------
    scene_id = tf.argmax(outputs["scene"][0]).numpy()
    scene_name = SCENE_CLASSES[scene_id]

    # -------- Detection --------
    det = outputs["detection"]
    boxes = det["boxes"][0].numpy()
    obj = det["objectness"][0].numpy().squeeze()
    cls = tf.argmax(det["class_logits"][0], axis=-1).numpy()

    # Resize image back
    img_vis = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # -------- Draw segmentation --------
    seg_resized = cv2.resize(
        seg_mask,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    overlay = img_vis.copy()
    for k, color in SEG_COLORS.items():
        overlay[seg_resized == k] = color

    img_vis = cv2.addWeighted(img_vis, 0.6, overlay, 0.4, 0)

    # -------- Draw boxes --------
    for i in range(len(boxes)):
        if obj[i] < OBJ_THRESH:
            continue

        x1, y1, x2, y2 = boxes[i]
        x1 = int(x1 * orig_w)
        x2 = int(x2 * orig_w)
        y1 = int(y1 * orig_h)
        y2 = int(y2 * orig_h)

        class_id = int(cls[i]) + 1
        label = CLASS_MAP.get(class_id, str(class_id))

        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            img_vis, label,
            (x1, max(y1-5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0,255,0), 2
        )

    cv2.putText(
        img_vis,
        f"Scene: {scene_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255,0,0), 2
    )

    return cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

# =========================
# TKINTER UI
# =========================

class App:
    def __init__(self, root):
        self.root = root
        root.title("Traffic Perception Demo")

        self.btn = tk.Button(
            root,
            text="Open Image",
            command=self.open_image,
            font=("Arial", 14)
        )
        self.btn.pack(pady=10)

        self.label = tk.Label(root)
        self.label.pack()

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        if not file_path:
            return

        image = Image.open(file_path).convert("RGB")
        result = run_inference(image)

        result_pil = Image.fromarray(result)
        result_pil.thumbnail((1000, 700))

        self.tk_img = ImageTk.PhotoImage(result_pil)
        self.label.config(image=self.tk_img)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
