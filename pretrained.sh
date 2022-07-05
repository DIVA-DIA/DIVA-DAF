CUDA_VISIBLE_DEVICES=5 python run.py experiment=cb55_full_run_deeplab_pt_vinay name=a &
CUDA_VISIBLE_DEVICES=6 python run.py experiment=cb55_full_run_deeplab_pt_vinay name=b &
CUDA_VISIBLE_DEVICES=7 python run.py experiment=cb55_full_run_deeplab_pt_vinay name=c &
CUDA_VISIBLE_DEVICES=5 python run.py experiment=cb55_full_run_deeplab_pt_vinay name=d &
CUDA_VISIBLE_DEVICES=6 python run.py experiment=cb55_full_run_deeplab_pt_vinay name=e & 
