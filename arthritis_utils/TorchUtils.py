import torch
from torch import nn

def device_check(volume):
    if (device := str(volume.device)) != "gpu":
        print(f"Warning: Tensor is on device {device}. Consider moving tensor to gpu first.")


def stack_tensors(outputs, name):
    tensor_stack = []
    for x in outputs:
        tensors = x[name]
        for tensor in tensors:
            tensor_stack.append(tensor)
    tensor_stack = torch.stack(tensor_stack)
    return tensor_stack


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x