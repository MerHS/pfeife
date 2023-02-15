# Pfeife: Automatic pipeline parallelism powered by TorchDynamo

## Prerequisite

- PyTorch (nightly version >= 2.0)
- torchbench (see below)

## Installation

```sh
# Clone this repo
git clone https://github.com/MerHS/pfeife

# Install PyTorch Benchmarks
## See https://github.com/pytorch/benchmark to install required Python packages
git clone https://github.com/pytorch/benchmark
cd benchmark
python install.py

# Run benchmarks
cd ../pfeife

## Single benchmark (VGG16, pipelined with aot_eager compiler)
python benchmark.py --model vgg16

## prints each step of the pipeline
python benchmark.py --model vgg16
```

## Benchmark Configuration

`benchmark.py` calculates an average time of a single training loop for torchbench models.
Also it can run the PyTorch profiler.

**Commandline argument**

```sh
Usage: python benchmark.py [...]

# Debugging
--no_pipe: Run without pipeline parallelism
--verbose: Print debugging information
--profile: Run PyTorch profiler. The result will be saved to ./result directory
--check_valid: Compare the result and gradient with an unsplitted CPU model

# Pipeline
--backend: Choose TorchDynamo compiler backend (default: aot_eager)
--scheduler: Select Pipeline Scheduler (gpipe, 1f1b, or bfs)
--device_cnt: Max count of GPU devices
--pipe_split: Number of splittion point of FX graph.
--batch_split: Number of microbatches

# Model & Training
--repeat: length of training loop
--batch_size: Size of minibatch
--model: Name of torchbench model (one of the directory name of https://github.com/pytorch/benchmark/tree/main/torchbenchmark/models)
```

## How to see profilling results

```sh
# Install tensorboard then
tensorboard --logdir=./result
# Then open https://localhost:6006
```
