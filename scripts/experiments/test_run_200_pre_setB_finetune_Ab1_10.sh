#!/usr/bin/env bash

set -e

weight="/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/sem_seg_synthetic_DPC_60_unet16_loss_no_weights/2022-11-08/17-47-06/checkpoints/epoch\=195/backbone.pth"

params_unet="experiment=fine_tune_cb55_AB1_train_20_run_unet16.yaml
        trainer.devices=[1]
        trainer.max_epochs=100
        trainer.accumulate_grad_batches=1
        datamodule.batch_size=5
        name=fine_tune_DPC60-SetB_cb55_AB1_training-10_run_unet16
        model.backbone.path_to_weights=${weight}
        +model.backbone.strict=False
        +metric.average=micro
        datamodule.train_folder_name=training-10
        logger.wandb.tags=[unet16,AB1,training-10,fine-tune,DPC60-SetB,4-classes,baseline,100-epochs,no-weights,best-jaccard,unet-adapted]
        logger.wandb.group=D60-FT-SetA-training-10"
python run.py ${params_unet}
