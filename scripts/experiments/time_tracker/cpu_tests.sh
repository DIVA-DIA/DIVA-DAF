#! /bin/bash

set -e

# python run.py experiment=time_taking_cifar_resnet50.yaml trainer.accelerator='cpu' trainer.devices=1 name="time_tracking_resnet50_cifar10_cpu_1"
# python run.py experiment=time_taking_cifar_resnet50.yaml trainer.accelerator='cpu' trainer.devices=2 name="time_tracking_resnet50_cifar10_cpu_2"
python run.py experiment=time_taking_cifar_resnet50.yaml trainer.accelerator='cpu' trainer.devices=4 name="time_tracking_resnet50_cifar10_cpu_4"
python run.py experiment=time_taking_cifar_resnet50.yaml trainer.accelerator='cpu' trainer.devices=8 name="time_tracking_resnet50_cifar10_cpu_8"
