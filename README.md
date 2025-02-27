<h3 align="center">
lsgpu is a light-weight python tool to retrieve GPU information
</h3>

### Install Dependency
```bash
pip install pycuda
```
### Download the source
```bash
wget https://raw.githubusercontent.com/liuy/lsgpu/refs/heads/master/lsgpu.py
```
### Run
```bash
python3 lsgpu.py
```
### Output
```
Found 2 CUDA-capable device(s)

GPU 0: NVIDIA GeForce RTX 4060 Ti
  Architecture: Ada Lovelace
  Compute capability: 8.9
  Number of SMs: 34
  CUDA cores per SM: 128
  Total CUDA cores: 4352
  Tensor Cores: 136 (4/SM, 4th gen)
  RT Cores: 34 (1/SM, 3rd gen)
  Max shared memory per block: 48.0 KB
  L2 cache size: 32.00 MB
  Clock rate: 2565.00 MHz
  Compute mode: Default

GPU 1: Tesla P40
  Architecture: Pascal
  Compute capability: 6.1
  Number of SMs: 30
  CUDA cores per SM: 128
  Total CUDA cores: 3840
  Tensor Cores: Not available (pre-Volta architecture)
  RT Cores: Not available
  Max shared memory per block: 48.0 KB
  L2 cache size: 3.00 MB
  Clock rate: 1531.00 MHz
  Compute mode: Default
```
