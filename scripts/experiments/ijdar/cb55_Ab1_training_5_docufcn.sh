#!/usr/bin/env bash

set -e

params_unet="experiment=cb55_AB1_train_20_run_docufcn.yaml
        trainer.devices=[3,4]
        +trainer.precision=32
        mode=ijdar.yaml
        name=sem_seg_baseline_cb55_AB1_loss_no_weights_doc_ufcn_800ep
        logger.wandb.tags=[doc-ufcn,AB1,4-classes,baseline,800-epochs,no-weights,training-10,time-measure]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=baseline_AB1-training-10-docufcn
        datamodule.train_folder_name=training-10
        trainer.max_epochs=800"

for i in {0..4}; do
  python run.py ${params_unet}
done
