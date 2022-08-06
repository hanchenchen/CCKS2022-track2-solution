export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 src/val.py -opt options/63_grad_clip_norm_0.5.yml