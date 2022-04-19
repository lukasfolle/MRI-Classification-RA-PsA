import torch
from torch import nn
from deep_learning.architectures.SAE.sae import CAE
from deep_learning.architectures.Resnet3D.models.resnet import generate_model


class CombinedModel(nn.Module):
    def __init__(self, number_of_classes=3, extends=[[0, 0, 0]]):
        super().__init__()
        self.weighting_models = nn.ModuleList()
        for extend in extends:
            self.weighting_models.append(generate_model(50, n_input_channels=1))
        self.activation1 = torch.nn.ReLU()
        self.fc1 = nn.Linear(len(extends) * 3, number_of_classes)


    def forward(self, data):
        y_hats = []
        for ind, d in enumerate(data):
            d = torch.transpose(d, 2, 4)
            y_hat = self.weighting_models[ind](d)
            y_hats.append(y_hat)
        y_hat = torch.cat(y_hats, dim=1)
        y_hat = self.activation1(y_hat)
        y_hat = self.fc1(y_hat)
        return y_hat


if __name__ == "__main__":
    import numpy as np

    cm = CombinedModel(extends=[[520, 520, 16], [520, 520, 16], [520, 520, 16], [320, 320, 64], [320, 320, 64]])
    x = (torch.from_numpy(np.random.rand(1, 1, 520, 520, 16).astype("float32")),
         torch.from_numpy(np.random.rand(1, 1, 520, 520, 16).astype("float32")),
         torch.from_numpy(np.random.rand(1, 1, 520, 520, 16).astype("float32")),
         torch.from_numpy(np.random.rand(1, 1, 320, 320, 64).astype("float32")),
         torch.from_numpy(np.random.rand(1, 1, 320, 320, 64).astype("float32")))
    prediction = cm(x)
    print(prediction)
