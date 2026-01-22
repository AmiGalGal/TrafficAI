data.py

load\_image - receive a path, return an image.

load\_json - receive a path, return json text.

extract\_detection\_objects - receiving a frame and returning an array of boxes, every box has x1,y1,x2,y2,label.

extract\_segmentation\_polygons - receiving a frame and returning an array of poly2d polygons.

poly2d\_to\_mpl\_patches - receiving a poly2d object and transitioning him into a patch.

testing - taking a photo and a labels and representing it.

checks - go through each of the labels directories to find missing labels, and returning the diff classes.



preprocess.py

resize\_img - resizing an image to another size.

normalize - normalization of a photo.

build\_segmentation\_mask - receive a list of poly2d object and return a 2d array mask\_h\*mask\_w that every pixel belong to a category.

visualize\_image\_and\_mask - visualize and image+its segmentation.

testing - showing a processed photo after resizing and normalization.

boxes to tensor - transforming boxes to a tensor.

preprocess\_sample - preprocess an image and its corresponding annotation.

tf\_preprocess - wraps preprocess\_sample in a TF compatible pipeline.

tf\_preprocess\_with\_augmentation - exactly like tf\_preprocess but with added augmentation.
preprocess\_image - preprocess PIL image for inference.



augmentations.py

random\_brightness - random brightness on a photo.

random\_contrast - random contrast on a photo.

random\_saturation - random saturation on a photo.

random\_hue - random hue on a photo.

random\_blur - random blur on a photo.

apply\_color\_augmentation - apply all the augmentation above on a photo.



tf\_pipeline.py

build\_dataset - builds a TF dataset.



visualizeBatch.py

resize\_boxes - rescales bounding boxes from original image size to target image size.

visualize\_batch\_from\_dataset - Visualizes a batch from a dataset with segmentation masks and bounding boxes overlaid.



backbone.py

build\_backbone - building the base model by receiving a model name (resnet50 / efiicientnetb0) and build it.



detectionHead.py

build\_detection\_head – builds a simple anchor-free detection head producing boxes, class logits, and objectness scores (not YOLO).



segmentationHead.py

build\_segmentation\_head – builds a lightweight semantic segmentation head and upsamples logits to fixed resolution.



sceneHead.py

build\_scene\_head – builds a scene classification head producing scene logits from backbone features.



multitask.py

build\_multitask\_model – builds a multi-task model with shared backbone and detection, segmentation, and scene heads.



metrics.py

box\_iou – computes pairwise Intersection over Union (IoU) between two sets of bounding boxes.

detection\_mean\_iou – computes mean IoU for valid detection boxes between predictions and ground truth.

segmentation\_pixel\_accuracy – computes pixel-wise accuracy for semantic segmentation, masked by validity.

segmentation\_mean\_iou – computes mean IoU across all segmentation classes.

scene\_accuracy – computes classification accuracy for scene prediction using a lookup table.



eval.py

list\_files – lists image files and matches them with corresponding JSON label files.

eval\_step – runs a forward pass and computes losses and metrics for one evaluation batch.

detection\_loss – computes combined box regression, classification, and objectness loss for detection.

segmentation\_loss – computes masked semantic segmentation loss.

scene\_loss – computes scene classification loss.

evaluation – runs full evaluation loop over dataset and prints averaged metrics.

main – loads evaluation dataset and model, then runs evaluation.



train.py

list\_files – lists image files and matches them with corresponding JSON label files.

detection\_loss – computes combined box regression, classification, and objectness loss for detection.

segmentation\_loss – computes masked semantic segmentation loss.

scene\_loss – computes scene classification loss.

train\_step – performs one training step with gradient computation, clipping, and optimizer update.

train – runs the full training loop with validation, TensorBoard logging, early stopping, and checkpointing.

validate\_step – runs a forward pass on validation data and computes losses and metrics.

main – builds datasets and model, trains heads first, then fine-tunes the full model.



demo.py

predictions\_to\_labels – converts model outputs into structured label format with segmentation mask, boxes, and scene.

single\_image\_dataset – creates a one-sample tf.data.Dataset from a single image and its labels















































