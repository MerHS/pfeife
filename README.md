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

## Benchmark commandline arguments

```
--model: name of torchbench model (default: timm_vision_transformer (a.k.a. ViT-S/16))
--backend: TorchDynamo backend (default: aot_eager)
--no_pipe: do not use pipeline parallelism
--verbose: prints every step of the pipeline
--check_valid: check the validity of training. it compares parameter, gradient, output of a single training loop with vanilla pytorch
--batch_size: batch size of a test input. must be a multiple of batch_split (default: depends on the model)
--device_cnt: number of usable GPU (default: 2)
--pipe_split: number of pipeline stages. must be a multiple of device_cnt (default: 2)
--batch_split: number of micro-batchs.
--repeat: length of a training loop
```
