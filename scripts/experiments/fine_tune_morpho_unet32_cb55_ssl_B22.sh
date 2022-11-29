#!/usr/bin/env bash
set -e

training_sets=("training-10"
  "training-20")
#  "training-40")

epochs=(100 100 50)

for i in "${!training_sets[@]}"; do
  devices="[6,7]"
  if [ "${training_sets[i]}" == "training-10" ]; then
    devices="[6]"
  fi
  params_unet="experiment=morpho_cb55_A11_unet32_finetune.yaml
        trainer.devices=${devices}
        trainer.max_epochs=${epochs[i]}
        name=morpho_fine_tune_AB1_${training_sets[i]}_cb55_B22_unet32_loss_no_weights
        model.backbone.path_to_weights=/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/morpho_cb55_B22_unet32_loss_no_weights/2022-11-11/10-49-27/checkpoints/epoch\=56/backbone.pth
        datamodule.train_folder_name=${training_sets[i]}
        logger.wandb.tags=[unet32,AB1,${training_sets[i]},fine-tune,Morpho,4-classes,baseline,${epochs[i]}-epochs,no-weights]
        logger.wandb.group=Morpho_B22-${training_sets[i]}"
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
