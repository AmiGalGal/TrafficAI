architecture of the multi task traffic model



overview

&nbsp;	the model is designed to perform 3 task at the same time

&nbsp;	1. object detection - detect cars, signs, lights, trucks etc

&nbsp;	2. segmentation - classify area who are drivable, lane, alt lane etc

&nbsp;	3. scene classification - where the photo is city street, highway etc



backbone

&nbsp;	architecture: Resnet50

&nbsp;	purpose: extracting feature maps out of input images

&nbsp;	input shape: (320x320x3)

&nbsp;	output: high level feature map

&nbsp;	in training: phase 1 - frozen, phase 2 active



detection head

&nbsp;	purpose: detection of objects with bounding boxes, classes and objectness score

&nbsp;	outputs: boxes - cords \[x1,y1,x2,y2], class\_logits - traffic class, objectness - score for each class

&nbsp;	losses: box regression - Huber loss L1, classification - sparce SoftMax cross entropy, objectness - binary cross entropy



segmentation head

&nbsp;	purpose: assign a class for every pixel in the photo

&nbsp;	output: logits for the 12 seg classes

&nbsp;	loss: sparse SoftMax cross entropy



scene classification head

&nbsp;	purpose: predict the env the of the image

&nbsp;	output: 7 logits for the 7 scene classes

&nbsp;	loss: sparse SoftMax cross entropy



multitask training

&nbsp;	total loss = 0.001\*detection\_loss + segmentation\_loss + scene\_loss

&nbsp;	training phases:

&nbsp;	phase 1: Heads only, the backbone is frozen.

&nbsp;	phase 2: fine tune the entire model, unfreeze the backbone, train all the layers

&nbsp;	optimization: Adam + learning rate schedule



metrics

&nbsp;	detection: box IoU, mIoU

&nbsp;	segmentation: pixel accuracy, mIoU

&nbsp;	scene classification: accuracy



dataflow

&nbsp;	input: jpg + json

&nbsp;	preprocess: resize -> normalize -> convert json to numerical lables

&nbsp;	forward pass: backbone -> 3 heads

&nbsp;	loss: compute detection, segmentation and scene losses

&nbsp;	backpropagation: update model weight

&nbsp;	validation: compute metrics on unseen data

&nbsp;	









&nbsp;	

