# pynvidia
A simple, lightweight Python library that provides NVIDIA GPU utility functions.

## Why?
I found it annoying that my continuous integration tests for deep learning default to using a fixed GPU. This default behaviour leads to problems when memory-intensive tests are being done on GPU 0 when GPU 0 is at 90% memory usage. The consequence is failed tests and, in turn, merge/pull requests that cannot pass because of `CUDA out of memory` errors.

## Solution
`pynvidia` is a simple Python library that parses and processes the command line output of `nvidia-smi` so that GPUs can be dynamically selected at runtime. Specifically, the least utilised GPU(s) can be selected for a machine learning job.

## Install me
```
$ pip install pynvidia
```

## Requirements
`nvidia-smi` must be installed on your system.

## Example usage
### Dynamically select the most appropriate GPU to use at runtime
```python
import os
import pynvidia as pyn

gpu_id = pyn.gpu_using_least_memory()

...

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
```

## API
### memory
Utilities for getting GPU memory usage.

- `pynvidia.memory.gpu_using_least_memory()`
- `pynvidia.memory.gpus_using_least_memory(num_gpus=1)` (future)

