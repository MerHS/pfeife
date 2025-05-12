# Pfeife: Automatic Pipeline Parallelism for PyTorch (ICML 2025)

Ho Young Jhoo, Chung-Kil Hur, Nuno P. Lopes.

## Prerequisite

- Python 3.10
- PyTorch 2.3.0
  - Starting from 2.4, `torch.compile` removes `call_module` nodes from the graph and all the trainable parameters are moved to fx function arguments. This should be fixed in the future.
- scipy
- networkx

To run the benchmarks, you need to install:

- torchbench (see below)
- datasets
- diffusers

## Installation & Usage

### 1. Install dependencies

```sh
git clone https://github.com/MerHS/pfeife
cd pfeife
pip install -r requirements.txt # install dependencies for reproducing the paper
```

### 2. Basic benchmarks

```sh
# Benchmark large models
bash bench_models/train_llama.sh
bash bench_models/train_sdxl.sh
bash bench_models/train_vit.sh
```

### 3. Install torchbench

```sh
# Install Torchbench
## See https://github.com/pytorch/benchmark to install required Python packages
cd ..
git clone https://github.com/pytorch/benchmark
cd benchmark
git checkout 1e7ed466 # tested from the paper

# >>> WARNING: pip dependency conflict <<<
## For some environments, `torch.__version__` contains cuda version appendix, e.g. `2.7.0+cu126`, that causes pip dependency conflict.
## To fix this, we need to manually edit `utils/__init__.py` to remove it.
##
## utils/__init__.py:L28
##  -        version = subprocess.check_output(cmd).decode().strip()
##  +        version = subprocess.check_output(cmd).decode().split("+")[0].strip()

python install.py
cd ../pfeife

# Run coverage & correcteness tests
bash coverage_test.sh

# Local benchmark / memcpy
python benchmark_local.py \
  --device_cnt=2 \
  --batch_size=16 \
  --batch_cnt=4 \
  --repeat=2 \
  --model timm_vision_transformer \
  --device_bench profile/stable-memcpy-dev2.bench

# Benchmark w/ NCCL + DP + FP16
python mp_launch.py --world_size 4 --dp_mesh "2,-2" benchmark_mp.py \
  --mixed_precision fp16 \
  --device_cnt=2 \
  --batch_size=2 \
  --batch_cnt=4 \
  --repeat=10 \
  --loop_cnt=2 \
  --model=timm_vision_transformer_large \
  --device_bench profile/stable-nccl-dev4.bench
```

## Command-line arguments

See `pfeife/option.py` for more details.

```sh
python mp_launch.py \
  --world_size 8 \ # Number of training devices
  --nnodes 2 \ # Number of nodes
  --node_rank 0 \ # Rank of the node (for distributed training)
  --dp_mesh "-2,4" \ # Structure of DP and PP mesh. Format: "-2,4" = 2 (neg: DP size) * 4 (pos: PP size).
  --master_addr localhost \ # Address of the master node
  --master_port 24567 \ # Port of the master node
    benchmark_mp.py \
      --device_cnt 4 \ # Number of devices for pipeline parallelism.
      --mixed_precision fp16 \ # Mixed precision training. fp16 or bf16
      --device_map "0,1,2,3" \ # Device id for PP mesh. If None, it will be set automatically.
      --batch_size 8 \ # *required* Size of a single micro-batch.
      --batch_cnt 8 \ # *required* Number of micro-batches for a single traning step.
      --loop_cnt 2 \ # How many times a looped pipeline will loop.
      --loop_batch 4 \ # How many micro-batches will be executed in a single loop.
      --prefetch_fwd 2 \ # Number of forward passes should be prefetched.
      --split_points "10,20,30,40,50,60,70" \ # Manual split points. If it is set, pfeife auto-splitting will be disabled.
      --device_bench profile/stable-nccl-dev4.bench \ # Path to save device communication benchmark results. You should set this option if you want to reuse device benchmark for multiple models and training.
      --graph_bench=./vit-g14-bs24 \ # Path to save graph benchmark results (i.e. memory usage and latency per operation). If you change the batch size, you must reset this value.
      --verbose # print verbose data
```

To use both DP and PP, you should set two integer values to `--dp_mesh` that indicates the size of PP devices and DP devices. For example, `--dp_mesh "-2,4"` means that there are 2 DP devices (negative) and 4 PP devices (postitive), total 8 devices.

The structure of DP-PP mesh is shown below.

```
--dp_mesh "-2,4": 2x4 matrix

  [
    [0, 1, 2, 3], # PP mesh 0
    [4, 5, 6, 7], # PP mesh 1
  ]

--dp_mesh "2,-4": 2x4 matrix

  [
    [0, 1, 2, 3], # DP mesh 0
    [4, 5, 6, 7], # DP mesh 1
  ]

--dp_mesh "-4,2": 4x2 matrix

  [
    [0, 1], # PP mesh 0
    [2, 3], # PP mesh 1
    [4, 5], # PP mesh 2
    [6, 7], # PP mesh 3
  ]
```

## How to train custom models

Please refer to the train scripts in `bench_models/models/` and benchmark codes in `benchmark_local.py` and `benchmark_mp.py`.

To train your custom models, you should execute the following command.

- `initialize_pfeife()` at the beginning of the script.
- Change the training loop like below:

```python
# Before
with batch from dataloader:
    optimizer.zero_grad() # deleted

    ##### should be wrapped #####
    outputs = model(pixel_values=batch["pixel_values"]).logits.view(-1, 10)
    labels = batch["labels"]
    loss = criterion(outputs, labels)
    loss.backward()
    ##### ----------------- #####

    optimizer.step() # deleted

# After
option = PipeOption.from_args(args) # or you can set the options manually
runner = PartialRunner(option, optimizer)
model = torch.compile(model, backend=partial_compiler)

with batch from dataloader:
    def iter_fn(batch):
        outputs = model(pixel_values=batch["pixel_values"]).logits.view(-1, 10)
        labels = batch["labels"]
        loss = criterion(outputs, labels)
        loss.backward()
        return loss

    runner.set_exec_fn(iter_fn)
    losses = runner.step(batch)
```

You can also use memcpy version rather than NCCL communication (memcpy is slightly faster than NCCL, but limited to single-node training). See `benchmark_local.py` for the exact usage.

## Anaylsis of the result

If you run a benchmark, you will see the result like below.

```
> python mp_launch.py --world_size 2 benchmark_partial.py --device_cnt=2 --batch_size=4 --batch_cnt=3 --repeat=2 --model=timm_vision_transformer_large --loop_cnt=1 --prefetch_fwd=0 --device_bench profile/stable-nccl-dev4.bench

================ run timm_vision_transformer_large ================
forward: 136.6731 ms, backward: 290.7007 ms
initial split points for 2 nodes: [205]
test PipeSched(node_cnt=2, loop_batch=3, prefetch_fwd=0)
beam 0 - 850.1693382244532 / [205] False - 850.1693382244532 / [210] False
beam 1 - 850.1693382244532 / [205] True - 850.1693382244532 / [210] True
elapsed time: 0.07s, result: 850.1693382244532
split points: [205], expect time: 850.1693382244532, schedule: PipeSched(node_cnt=2, loop_batch=3, prefetch_fwd=0)
dev 0 - idle mem: 7797.13MB peak: 11530.17MB
dev 1 - idle mem: 7826.63MB peak: 9694.14MB
GRAPH: 2 nodes / 1 edges
==== nodes ====
Node [0/0] (weight: 7723.98MB, act_mem: 1860.00MB, temp_mem: 1873.04MB, comp_time: 67.84899202510715, back_time: 144.31329663665215, optim_time: 0, node len: 205, send weight: 11.04MB)
Node [1/1] (weight: 7727.24MB, act_mem: 1854.48MB, temp_mem: 1867.51MB, comp_time: 67.92560639008879, back_time: 144.47625368656873, optim_time: 0, node len: 201, send weight: 0.00MB)
==== edges ====
Edge (0 -> 1) (weight: 11.04MB, device: 0 -> 1, weight: 11.04MB, send time: 0.2602601869048874)
mean latency 0.8966332444979344 / std.dev. 0.0009473134996369481 across 2 runs
(06:44:22.211) [Worker 1] memory usage: idle: 7899.59MB, max: 9904.68MB, alloc: 2005.09MB, reserved: 10940.00MB
(06:44:22.211) [Worker 0] memory usage: idle: 7914.72MB, max: 11693.23MB, alloc: 3778.52MB, reserved: 12634.00MB
```

It will print the shape and memory profiling result of the fused graph, and the memory usage of each node.
Here, we have 2 nodes which are connected like `0 -> 1`.

It will print each node's weight (total bytes of param + gradient + optimizer state), saved activation memory of a single mini-batch from the forward pass, usage of temporary memory while running the node (dealloced right before the forward pass is done and the activation memory is allocated), sum of computation time of the FX operations inside the node, and the number of computation nodes.

Then, it prints mean latency of a single pipeline execution and the memory usage of each device.

- `idle`: Memory allocation of the idle state. Should be equal to the total bytes of assigned weights and input values.
- `max`: Maximum memory allocation through the while pipeline.
- `alloc`: `max` - `idle`. Should be similar to the maximum size of the allocated activation memory with the effects of temporary memory.
- `reserved`: PyTorch reserved memory of the device.

## Troubleshooting

- Restart error after force kill (i.e. Ctrl+C).
  - Try `pkill -9 pt_main_thread` to kill the zombie process.
