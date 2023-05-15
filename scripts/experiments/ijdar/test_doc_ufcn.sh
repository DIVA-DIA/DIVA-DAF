#!/usr/bin/env bash

set -e

params_unet="experiment=fine_tune_csg18_new_split_run_doc_ufcn.yaml
        test=true
        train=false
        trainer.devices=[0]
        +trainer.precision=32
        mode=ijdar.yaml
        name=fine_tune_csg18_run_doc_ufcn-100ep-validation-test
        +model.backbone.path_to_weights=/net/research-hisdoc/model_weights/doc-ufcn/iou_Doc_UFCN_CSG18_100epochs_mix_synt_120-60_finetuned_100epochs-just_weights.pt
        logger.wandb.tags=[doc_ufcn,AB1,test,4-classes,baseline,no-weights,best-jaccard]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=csg18-validation-test-100ep"
#      echo ${params_unet}
python run.py ${params_unet}
