#!/usr/bin/env bash
set -e

training_sets=("training-10")
#  "training-20")
#  "training-40")

for set in ${training_sets[*]}; do
  devices="[6,7]"
  if [ "${training_sets[i]}" == "training-10" ]; then
    devices="[6]"
  fi
  params_unet="experiment=morpho_cb55_A11_unet_finetune.yaml
        trainer.devices=${devices}
        trainer.max_epochs=100
        name=morpho_fine_tune_AB1_${set}_cb55_B22_unet_loss_no_weights
        model.backbone.path_to_weights=/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/morpho_cb55_B22_unet_loss_no_weights/2022-11-11/11-16-58/checkpoints/epoch\=46/backbone.pth
        datamodule.train_folder_name=${set}
        logger.wandb.tags=[unet,AB1,${set},fine-tune,Morpho,4-classes,baseline,100-epochs,no-weights]
        logger.wandb.group=Morpho_B22-${set}"
  python run.py ${params_unet}
done

#
#for i in ${1..2}
#do
#  for arch in ${archs[*]}
#  do
#    params="experiment=morpho_cb55_A11_unet_finetune.yaml
#    trainer.devices=[4,5,6,7]
#    logger.wandb.tags=[best_model,Morpho,pre-training,${arch},B22-60,CB55-ssl-set,200-epochs]"
#    python run.py ${params}
#  done
#done
