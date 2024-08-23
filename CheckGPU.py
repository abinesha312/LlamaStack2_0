import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. ")
else:
    raise RuntimeError("Running ")