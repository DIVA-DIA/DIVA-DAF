CUDA_VISIBLE_DEVICES=5 python run.py experiment=cb55_full_run_deeplab_pt_vinay name=0 &
CUDA_VISIBLE_DEVICES=6 python run.py experiment=cb55_full_run_deeplab_pt_vinay name=1 &
CUDA_VISIBLE_DEVICES=7 python run.py experiment=cb55_full_run_deeplab_pt_vinay name=2 &
CUDA_VISIBLE_DEVICES=5 python run.py experiment=cb55_full_run_deeplab_pt_vinay name=3 &
CUDA_VISIBLE_DEVICES=6 python run.py experiment=cb55_full_run_deeplab_pt_vinay name=4 & 
