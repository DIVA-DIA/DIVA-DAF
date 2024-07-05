#!/bin/bash

set -e

python run.py experiment=time_taking_imagenet_resnet50.yaml name="time_tracking_resnet50_imagenet_resized_gpu_1" trainer.accelerator="gpu" trainer.devices="[1]"
python run.py experiment=time_taking_imagenet_resnet50.yaml name="time_tracking_resnet50_imagenet_resized_gpu_2" trainer.accelerator="gpu" trainer.devices="[1,2]"
python run.py experiment=time_taking_imagenet_resnet50.yaml name="time_tracking_resnet50_imagenet_resized_cpu_1" trainer.accelerator="cpu" trainer.devices="1"
python run.py experiment=time_taking_imagenet_resnet50.yaml name="time_tracking_resnet50_imagenet_resized_cpu_2" trainer.accelerator="cpu" trainer.devices="2"
python run.py experiment=time_taking_imagenet_resnet50.yaml name="time_tracking_resnet50_imagenet_resized_cpu_4" trainer.accelerator="cpu" trainer.devices="4"
python run.py experiment=time_taking_imagenet_resnet50.yaml name="time_tracking_resnet50_imagenet_resized_cpu_8" trainer.accelerator="cpu" trainer.devices="8"
