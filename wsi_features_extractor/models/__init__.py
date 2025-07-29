import torch
import timm
import numpy as np
from torchvision import transforms

__all__ = ['list_models', 'get_model', 'get_custom_transformer']

__implemented_models = {
    'ctranspath': 'models/ckpts/ctranspath.pth',
    'gpfm': 'models/ckpts/GPFM.pth',
    'mstar': 'models/ckpts/mSTAR.pth',
    'conch15': 'models/ckpts/conch1.5.bin',
    'litepath-ti': 'models/ckpts/litepath-ti.pth',
    'omiclip': 'models/ckpts/omiclip.pth',
    'patho_clip': 'models/ckpts/Patho-CLIP-L.pt',
}


def list_models():
    print('The following are implemented models:')
    for k, v in __implemented_models.items():
        print('{}: {}'.format(k, v))
    return __implemented_models


def get_model(model_name, device, gpu_num, jit=False):
    """_summary_

    Args:
        model_name (str): the name of the requried model
        device (torch.device): device, e.g. 'cuda'
        gpu_num (int): the number of GPUs used in extracting features

    Raises:
        NotImplementedError: if the model name does not exist

    Returns:
        nn.Module: model
    """
    from models.uni2 import get_uni_model
    model = get_uni_model(device)
    return model


def get_custom_transformer(model_name):
    """_summary_

    Args:
        model_name (str): the name of model

    Raises:
        NotImplementedError: not implementated

    Returns:
        torchvision.transformers: the transformers used to preprocess the image
    """
    from models.uni2 import get_uni_trans
    custom_trans = get_uni_trans()
    return custom_trans
