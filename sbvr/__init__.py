from .core import (sbvr, sbvr_serialized, load, mm_T)
from sbvr.sbvr_cuda import _sbvr_cuda_init, _sbvr_mm_T
import torch

torch.serialization.add_safe_globals([sbvr_serialized])
_sbvr_cuda_init()