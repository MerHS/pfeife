# Pfeife: Automatic pipeline parallelism powered by TorchDynamo

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
python benchmark.py --model vgg16 --verbose
```
