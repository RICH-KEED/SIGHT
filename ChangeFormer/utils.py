import numpy as np
import torch
from torchvision import utils


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    
    # Check if CUDA is available and GPU IDs are valid
    if len(args.gpu_ids) > 0 and torch.cuda.is_available():
        try:
            torch.cuda.set_device(args.gpu_ids[0])
        except (AttributeError, RuntimeError):
            # Fallback to CPU if CUDA setup fails
            print("Warning: CUDA setup failed, falling back to CPU")
            args.gpu_ids = []
    else:
        # Use CPU if no GPU IDs or CUDA not available
        args.gpu_ids = []