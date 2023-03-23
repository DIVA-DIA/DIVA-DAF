#!/usr/bin/env bash

set -e

# 25 epochs
#weights=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-07/11-26-14/checkpoints/epoch\=20/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-07/11-35-28/checkpoints/epoch\=20/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-07/11-44-43/checkpoints/epoch\=21/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-07/11-53-50/checkpoints/epoch\=15/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-07/12-03-07/checkpoints/epoch\=23/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-07/12-12-47/checkpoints/epoch\=22/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-07/12-22-24/checkpoints/epoch\=22/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-07/12-31-59/checkpoints/epoch\=23/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-07/12-41-18/checkpoints/epoch\=21/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-07/12-50-51/checkpoints/epoch\=20/backbone.pth")

# 50 epcohs
weights=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-08/17-21-20/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-08/17-32-38/checkpoints/epoch\=47/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-08/17-43-50/checkpoints/epoch\=37/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-08/17-54-55/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-08/18-06-11/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-08/18-17-22/checkpoints/epoch\=47/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-08/18-28-43/checkpoints/epoch\=46/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-08/18-40-02/checkpoints/epoch\=47/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-08/18-51-24/checkpoints/epoch\=32/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights/2023-01-08/19-02-34/checkpoints/epoch\=35/backbone.pth")

# 5 epochs
#weights=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_5epoch/2023-01-09/10-50-24/checkpoints/epoch\=4/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_5epoch/2023-01-09/10-52-23/checkpoints/epoch\=4/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_5epoch/2023-01-09/10-54-21/checkpoints/epoch\=4/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_5epoch/2023-01-09/10-56-16/checkpoints/epoch\=4/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_5epoch/2023-01-09/10-58-16/checkpoints/epoch\=3/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_5epoch/2023-01-09/11-00-14/checkpoints/epoch\=4/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_5epoch/2023-01-09/11-02-09/checkpoints/epoch\=4/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_5epoch/2023-01-09/11-04-07/checkpoints/epoch\=4/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_5epoch/2023-01-09/11-06-05/checkpoints/epoch\=4/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_5epoch/2023-01-09/11-07-59/checkpoints/epoch\=3/backbone.pth")

# 100 epochs
#weights=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_100epoch/2023-01-11/10-15-36/checkpoints/epoch\=95/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_100epoch/2023-01-11/10-36-38/checkpoints/epoch\=95/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_100epoch/2023-01-11/10-58-05/checkpoints/epoch\=84/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_100epoch/2023-01-11/11-19-25/checkpoints/epoch\=81/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_unet_loss_no_weights_100epoch/2023-01-11/11-40-34/checkpoints/epoch\=96/backbone.pth")

for j in ${weights[*]}; do
  params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml trainer.devices=[4,5,6,7] +model.backbone.path_to_weights=${j}"
  python run.py ${params}
done