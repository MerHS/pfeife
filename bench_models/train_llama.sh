export PYTHONPATH="$PYTHONPATH:$(pwd)"

python mp_launch.py --world_size 8 bench_models/models/train_llama.py \
  --config_name 'enoch/llama-7b-hf' \
  --tokenizer_name 'enoch/llama-7b-hf' \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --block_size 256 \
  --per_device_train_batch_size 1 \
  --device_cnt 8 \
  --batch_cnt 8 \
  --loop_cnt 2