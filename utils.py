import torch

def as_tensor(*args):
    return tuple([torch.tensor(i) if not torch.is_tensor(i) else i for i in args ])