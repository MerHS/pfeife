export PYTHONPATH="$PYTHONPATH:$(pwd)"

python mp_launch.py --world_size 2 bench_models/models/train_vit.py \
  --device_bench profile/nccl-dev2.bench \
  --device_cnt 2 \
  --batch_size 24 \
  --batch_cnt 2
