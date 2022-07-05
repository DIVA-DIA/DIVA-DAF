CUDA_VISIBLE_DEVICES=0 python run.py experiment=cb55_full_run_deeplab_vinay name=a &
CUDA_VISIBLE_DEVICES=1 python run.py experiment=cb55_full_run_deeplab_vinay name=b &
CUDA_VISIBLE_DEVICES=2 python run.py experiment=cb55_full_run_deeplab_vinay name=c &
CUDA_VISIBLE_DEVICES=3 python run.py experiment=cb55_full_run_deeplab_vinay name=d &
CUDA_VISIBLE_DEVICES=4 python run.py experiment=cb55_full_run_deeplab_vinay name=e & 
