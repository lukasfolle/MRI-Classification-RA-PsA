# Adopted from https://github.com/kenshohara/3D-ResNets-PyTorch

import torch
from torch import nn
import sys
sys.path.insert(0, "/Documents/arthritis")

from deep_learning.architectures.Resnet3D.models import resnet2p1d, wide_resnet, pre_act_resnet, densenet, resnet


def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})

    return parameters


def generate_model(opt):
    assert opt["model"] in [
        'resnet', 'resnet2p1d', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    if opt["model"] == 'resnet':
        model = resnet.generate_model(model_depth=opt["model_depth"],
                                      n_classes=opt["n_classes"],
                                      n_input_channels=opt["n_input_channels"],
                                      shortcut_type=opt["resnet_shortcut"],
                                      conv1_t_size=opt["conv1_t_size"],
                                      conv1_t_stride=opt["conv1_t_stride"],
                                      no_max_pool=opt["no_max_pool"],
                                      widen_factor=opt["resnet_widen_factor"])
    elif opt["model"] == 'resnet2p1d':
        model = resnet2p1d.generate_model(model_depth=opt["model_depth"],
                                          n_classes=opt["n_classes"],
                                          n_input_channels=opt["n_input_channels"],
                                          shortcut_type=opt["resnet_shortcut"],
                                          conv1_t_size=opt["conv1_t_size"],
                                          conv1_t_stride=opt["conv1_t_stride"],
                                          no_max_pool=opt["no_max_pool"],
                                          widen_factor=opt["resnet_widen_factor"])
    elif opt["model"] == 'wideresnet':
        model = wide_resnet.generate_model(
            model_depth=opt["model_depth"],
            k=opt["wide_resnet_k"],
            n_classes=opt["n_classes"],
            n_input_channels=opt["n_input_channels"],
            shortcut_type=opt["resnet_shortcut"],
            conv1_t_size=opt["conv1_t_size"],
            conv1_t_stride=opt["conv1_t_stride"],
            no_max_pool=opt["no_max_pool"])
    elif opt["model"] == 'preresnet':
        model = pre_act_resnet.generate_model(
            model_depth=opt["model_depth"],
            n_classes=opt["n_classes"],
            n_input_channels=opt["n_input_channels"],
            shortcut_type=opt["resnet_shortcut"],
            conv1_t_size=opt["conv1_t_size"],
            conv1_t_stride=opt["conv1_t_stride"],
            no_max_pool=opt["no_max_pool"])
    elif opt["model"] == 'densenet':
        model = densenet.generate_model(model_depth=opt["model_depth"],
                                        num_classes=opt["n_classes"],
                                        n_input_channels=opt["n_input_channels"],
                                        conv1_t_size=opt["conv1_t_size"],
                                        conv1_t_stride=opt["conv1_t_stride"],
                                        no_max_pool=opt["no_max_pool"])

    return model


def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'])
        tmp_model = model
        if model_name == 'densenet':
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)
        else:
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)

    return model


def get_pretrained_resnet(depth=50, pretrained=True):
    if depth not in (available_sizes := [18, 34, 50, 200]):
        raise NotImplementedError(f"Only resnet {available_sizes} are currently downloaded.")
    opt = {"model": "resnet",
           "model_depth": depth,
           "n_classes": 700,
           "n_input_channels": 3,
           "resnet_shortcut": "B",
           "conv1_t_size": 7,
           "conv1_t_stride": 1,
           "no_max_pool": False,
           "resnet_widen_factor": 1.0
    }
    model = generate_model(opt)
    if pretrained:
        model = load_pretrained_model(model, f"/Documents/arthritis/deep_learning/architectures/Resnet3D/pretrained_checkpoints/r3d{depth}_K_200ep.pth", "resnet", 3)
    model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
    return model


if __name__ == "__main__":
    model = get_pretrained_resnet(18)
    test = torch.rand((1, 1, 64, 64, 64)).float()
    output = model(test)
    print(f"Output shape: {output.shape}")