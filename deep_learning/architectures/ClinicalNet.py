import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/Documents/arthritis")


class ClinicalNet(torch.nn.Module):
    def __init__(self, number_of_clinical_features=8, number_of_classes=2, drop_prob=0.2):
        super().__init__()
        self.clinical_model = torch.nn.Sequential(
            torch.nn.Linear(number_of_clinical_features, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(drop_prob),
            torch.nn.Linear(50, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(drop_prob),
            torch.nn.Linear(50, number_of_classes))

    def __call__(self, data):
        clinical_data = self.clean_clinical_data(data)
        prediction = self.clinical_model(clinical_data)
        return prediction

    @staticmethod
    def clean_clinical_data(clinical_data):
        clinical_data = torch.stack(clinical_data, dim=1)
        clinical_data = clinical_data.to(torch.float)
        return clinical_data


if __name__ == "__main__":
    net = ClinicalNet(number_of_clinical_features=3)
    import numpy as np

    pred = net((torch.from_numpy(np.array([0.02]).astype("float32")),
                torch.from_numpy(np.array([1.0]).astype("float32")),
                torch.from_numpy(np.array([0.0]).astype("float32"))))
    print(pred.shape)
    print(pred)
