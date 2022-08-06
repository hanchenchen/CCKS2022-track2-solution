export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4444 src/train.py -opt options/63_grad_clip_norm_0.5.yml