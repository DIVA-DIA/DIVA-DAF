#!/usr/bin/env bash
set -e

archs=("unet16"
       "unet32"
       "unet64"
       "unet")

for i in {1..2}
do
  for arch in ${archs[*]}
  do
    params="experiment=morpho_cb55_B22_${arch}.yaml
    trainer.devices=[0,1,2,3]
    name=morpho_cb55_B22_${arch}_loss_no_weights
    trainer.max_epochs=200
    logger.wandb.tags=[best_model,Morpho,pre-training,${arch},B22-60,CB55-ssl-set,200-epochs]"
    python run.py ${params}
  done
done