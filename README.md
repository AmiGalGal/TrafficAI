\# Traffic Multi-Task Model



This repository contains a multi-task deep learning model for traffic analysis:

\- Object Detection: detects cars, traffic lights, signs, trucks etc.

\- Segmentation: drivable areas, lanes, crosswalks.

\- Scene Classification: city street, highway, residential, etc.



The model is implemented in TensorFlow 2.13.0x.



---



\## Features



\- ResNet50 backbone (optionally frozen)

\- Multi-head outputs (detection, segmentation, scene)

\- Training with custom datasets

\- Real-time inference demo

\- Logging with TensorBoard



---



\## Installation



git clone <repo>

cd TrafficProject

pip install -r requirements.txt

Due to time and computation constraints train.py may be flawed.
