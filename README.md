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

## Run benchmark (VGG16, pipelined with aot_eager compiler)
python benchmark.py --model vgg16 --verbose

## Single process benchmark (run on a local machine)
python benchmark.py --model vgg16 --single_proc
```
