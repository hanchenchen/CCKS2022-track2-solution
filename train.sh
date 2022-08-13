export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4444 src/train.py -opt options/97_fp16_b63.yml