import os
import random
import datetime
import numpy as np
import torch

global_device = None
n_gpu:int = -1

def create_experiment_folder(model_output_dir, name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    exp = "{}__{}".format(name, timestamp)

    out_path = os.path.join(model_output_dir, exp)
    os.makedirs(out_path, exist_ok=True)

    return exp, out_path

def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if global_device is None or n_gpu == -1:
        setup_device()
    if global_device == 'cuda' and n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def setup_device():
    global global_device, n_gpu
    if torch.cuda.is_available():
        device_name = "cuda"
        n_gpu = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device_name = "mps"
        n_gpu = 1
    else:
        device_name = "cpu"
        n_gpu = 0
    global_device = torch.device(device_name)

    print("We use the device: '{}' and {} GPUs.".format(global_device, n_gpu))

    return global_device, n_gpu

def get_device():
    if global_device is None:
        setup_device()
    return global_device