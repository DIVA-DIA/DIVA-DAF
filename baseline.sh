CUDA_VISIBLE_DEVICES=0 python run.py experiment=cb55_full_run_deeplab_vinay name=0 &
CUDA_VISIBLE_DEVICES=1 python run.py experiment=cb55_full_run_deeplab_vinay name=1 &
CUDA_VISIBLE_DEVICES=2 python run.py experiment=cb55_full_run_deeplab_vinay name=2 &
CUDA_VISIBLE_DEVICES=3 python run.py experiment=cb55_full_run_deeplab_vinay name=3 &
CUDA_VISIBLE_DEVICES=4 python run.py experiment=cb55_full_run_deeplab_vinay name=4 & 
