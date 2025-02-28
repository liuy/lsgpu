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
  Compute Capability: 8.9
  Number of SMs: 34
  CUDA Cores per SM: 128
  Total CUDA Cores: 4352
  Tensor Cores: 136 (4/SM, 4th gen)
  RT Cores: 34 (1/SM, 3rd gen)
  Max Shared Memory per Block: 48.0 KB
  L2 Cache Size: 32.00 MB
  Clock Rate: 2565.00 MHz
  Memory Type: GDDR6X
  Memory Size: 15.60 GB
  Memory Bus Width: 128-bit
  Effective Memory Clock Rate: 9001.00 MHz
  Memory Speed: 288.0 GB/s
  Compute Mode: Default

GPU 1: Tesla P40
  Architecture: Pascal
  Compute Capability: 6.1
  Number of SMs: 30
  CUDA Cores per SM: 128
  Total CUDA Cores: 3840
  Tensor Cores: Not available
  RT Cores: Not available
  Max Shared Memory per Block: 48.0 KB
  L2 Cache Size: 3.00 MB
  Clock Rate: 1531.00 MHz
  Memory Type: GDDR5X
  Memory Size: 23.87 GB
  Memory Bus Width: 384-bit
  Effective Memory Clock Rate: 3615.00 MHz
  Memory Speed: 347.0 GB/s
  Compute Mode: Default
```
