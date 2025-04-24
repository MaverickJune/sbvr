from .core import (_sbvr_serialized, load, mm_T)
import sbvr.sbvr_cuda
import torch

torch.serialization.add_safe_globals([_sbvr_serialized])
sbvr.sbvr_cuda.sbvr_cuda_init()