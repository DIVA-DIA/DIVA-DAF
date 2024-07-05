#!/usr/bin/env bash

set -e

#gpu
python run.py experiment=time_taking_hisdb_patch_unet.yaml trainer.devices="[1]" name="time_tracking_hisdb_cropped_unet_gpu_1"
python run.py experiment=time_taking_hisdb_patch_unet.yaml trainer.devices="[1,2]" name="time_tracking_hisdb_cropped_unet_gpu_2"

# cpu
python run.py experiment=time_taking_hisdb_patch_unet.yaml trainer.devices="1" trainer.accelerator="cpu" name="time_tracking_hisdb_cropped_unet_cpu_1"
python run.py experiment=time_taking_hisdb_patch_unet.yaml trainer.devices="2" trainer.accelerator="cpu" name="time_tracking_hisdb_cropped_unet_cpu_2"
python run.py experiment=time_taking_hisdb_patch_unet.yaml trainer.devices="4" trainer.accelerator="cpu" name="time_tracking_hisdb_cropped_unet_cpu_4"
python run.py experiment=time_taking_hisdb_patch_unet.yaml trainer.devices="8" trainer.accelerator="cpu" name="time_tracking_hisdb_cropped_unet_cpu_8"

