#!/usr/bin/env bash

set -e

for i in {1..3}; do
  params="experiment=ssltiles_resnet50_cb55_prebuilt_960_1344.yaml
          trainer.max_epochs=40
          trainer.devices=[1,2,3,4]
          +trainer.check_val_every_n_epoch=5
          +callbacks.model_checkpoint.every_n_epochs=10
          mode=hip_tiles.yaml
          name=PT_cb55_3_fixed_resnet50_40epochs
          logger.wandb.project=hip-tiles
          logger.wandb.tags=[best_model,tiles,cb55,pre-training,resnet50,dataset1,40-epochs]
          logger.wandb.group=pt-tiles-cb55-resnet50-40-ep"
  python run.py ${params}
done
