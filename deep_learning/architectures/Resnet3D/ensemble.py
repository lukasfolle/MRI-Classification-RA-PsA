import torch
import torch.nn
from collections import OrderedDict

from deep_learning.architectures.Resnet3D.model import get_pretrained_resnet
from deep_learning.architectures.ClinicalNet import ClinicalNet


def load_trained_model(model, weights_path):
    print('loading pretrained model {}'.format(weights_path))
    pretrain = torch.load(weights_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in pretrain['state_dict'].items():
        name = k[6:] # remove `model.`
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    return model


class Ensemble(torch.nn.Module):
    def __init__(self, indices: list, fold: int, combined_clinical=False):
        super().__init__()
        self.imaging_models = torch.nn.ModuleList()
        self.clinical_model = None
        if combined_clinical:
            self.clinical_model = ClinicalNet()
        # ["T1_COR_agent_None", "T1_FS_COR_agent_GD", "T2_FS_COR_agent_None", "T1_FS_AX_agent_GD", "T2_FS_AX_agent_None"]
        # Fold 0
        if fold == 0:
             trained_paths = [
                 # Specify path to weights
                 "/path/to/weights/T1_COR_agent_None.ckpt",
                 "/path/to/weights/T1_FS_COR_agent_GD.ckpt",
                 "/path/to/weights/T2_FS_COR_agent_None.ckpt",
                 "/path/to/weights/T1_FS_AX_agent_GD.ckpt",
                 "/path/to/weights/T2_FS_AX_agent_None.ckpt",
                 # Add path to trained clinical model...
            ]
        # Fold 1
        if fold == 1:
             trained_paths = [
                 # Specify path to weights
                 "/path/to/weights/T1_COR_agent_None.ckpt",
                 "/path/to/weights/T1_FS_COR_agent_GD.ckpt",
                 "/path/to/weights/T2_FS_COR_agent_None.ckpt",
                 "/path/to/weights/T1_FS_AX_agent_GD.ckpt",
                 "/path/to/weights/T2_FS_AX_agent_None.ckpt",
            ]
        # Fold 2
        if fold == 2:
             trained_paths = [
                 # Specify path to weights
                 "/path/to/weights/T1_COR_agent_None.ckpt",
                 "/path/to/weights/T1_FS_COR_agent_GD.ckpt",
                 "/path/to/weights/T2_FS_COR_agent_None.ckpt",
                 "/path/to/weights/T1_FS_AX_agent_GD.ckpt",
                 "/path/to/weights/T2_FS_AX_agent_None.ckpt",
            ]
        # Fold 3
        if fold == 3:
             trained_paths = [
                 # Specify path to weights
                 "/path/to/weights/T1_COR_agent_None.ckpt",
                 "/path/to/weights/T1_FS_COR_agent_GD.ckpt",
                 "/path/to/weights/T2_FS_COR_agent_None.ckpt",
                 "/path/to/weights/T1_FS_AX_agent_GD.ckpt",
                 "/path/to/weights/T2_FS_AX_agent_None.ckpt",
            ]
        # Fold 4
        if fold == 4:
             trained_paths = [
                 # Specify path to weights
                 "/path/to/weights/T1_COR_agent_None.ckpt",
                 "/path/to/weights/T1_FS_COR_agent_GD.ckpt",
                 "/path/to/weights/T2_FS_COR_agent_None.ckpt",
                 "/path/to/weights/T1_FS_AX_agent_GD.ckpt",
                 "/path/to/weights/T2_FS_AX_agent_None.ckpt",
            ]
        trained_paths = [path for i, path in enumerate(trained_paths) if i in indices]
        for path in trained_paths:
            model = get_pretrained_resnet(34)
            model = load_trained_model(model, path)
            # Freeze all models
            for parameter in model.parameters():
                parameter.requires_grad = False
            self.imaging_models.append(model)
        
    
    def forward(self, x, clinical_data=None):
        predictions = []
        for x_, model in zip(x, self.imaging_models):
            pred = model(x_)
            predictions.append(pred)
        if self.clinical_model:
            predictions.append(self.clinical_model(clinical_data))
        ensemble_prediction = torch.mean(torch.stack(predictions, dim=0), 0)
        return ensemble_prediction
        