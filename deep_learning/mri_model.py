import json
import os
import shutil
import sys
from copy import deepcopy
import random

import numpy as np
import pytorch_lightning as pl
import json
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report, roc_curve
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from captum.attr import GuidedGradCam

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from deep_learning.architectures.Resnet3D.model import get_pretrained_resnet
from deep_learning.architectures.Resnet3D.ensemble import Ensemble
from arthritis_utils.TorchUtils import stack_tensors
from arthritis_utils.General import build_processing_pipeline, get_persistent_dataset, ArthritisDataLoader


def get_weighting_dim(weighting: str):
    extends = {"T1_COR_agent_None": [512, 512, 16],
               "T1_FS_COR_agent_GD": [512, 512, 16],
               "T2_FS_COR_agent_None": [512, 512, 16],
               "T1_FS_AX_agent_GD": [320, 320, 64],
               "T2_FS_AX_agent_None": [320, 320, 64]
               }
    return extends[weighting]


class DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.batch_size = hparams["batch_size"]
        self.dataset = None
        self.split = [0.7, 0.2, 0.1]
        self.db_length = 0
        self.num_workers = 12
        self.hparams = hparams
        self.transformation_settings = {}
        self.transformations = self.get_transformations()
        self.transformation_hparams = json.loads(
            json.dumps(self.transformations, default=lambda o: '<not serializable>'))
        self.preprocessing_pipeline = build_processing_pipeline(self.transformations)
        self.base_path = "/path/to/mri_data"
        self.clinic_db_path = "/path/to/clinical_data.xlsx"
        i = 0
        while True:
            if not os.path.exists((cache_dir := f"/cache_{i}/")):
                os.mkdir(cache_dir)
                break
            i += 1
        self.cache_dir = cache_dir

    def setup(self, stage=None):
        cache_monai_db_format = False
        # Database needs to be implemented for this to work and depends on the way the data lies on the drives
        db = Database(self.base_path, self.clinic_db_path, cache_monai_db_format=cache_monai_db_format)
        self.dataset = get_persistent_dataset(db, transforms=self.preprocessing_pipeline, cache_folder=self.cache_dir)
        print("INFO: Keeping only classes 0 and 1.")
        self.dataset.data = [sample for sample in self.dataset.data if sample["Class"] in [0, 1]]
        removed_missing_sequences = []
        self.additional_data_not_all_sequences = []
        for sample in self.dataset.data:
            drop = False
            for weighting in ["T1_COR_agent_None", "T1_FS_COR_agent_GD", "T2_FS_COR_agent_None", "T1_FS_AX_agent_GD", "T2_FS_AX_agent_None"]:
                if sample[weighting] is None:
                    drop = True
                    break
            if drop:
                if len(self.hparams["weightings"]) == 1 and sample[self.hparams["weightings"][0]] is not None:
                    self.additional_data_not_all_sequences.append(sample)
            if not drop:
                removed_missing_sequences.append(sample)
        self.dataset.data = removed_missing_sequences
        random.seed(42)
        random.shuffle(self.dataset.data)
        self.db_length = len(self.dataset)

    def get_transformations(self):
        vc_first_keys = [key for key in self.hparams["weightings"] if
                         key in ["T1_FS_COR_agent_GD", "T1_FS_AX_agent_GD"]]
        vc_second_keys = [key for key in self.hparams["weightings"] if key in ["T2_FS_AX_agent_None"]]
        vc_third_keys = [key for key in self.hparams["weightings"] if
                         key in ["T1_COR_agent_None", "T2_FS_COR_agent_None"]]
        cor_keys = [key for key in self.hparams["weightings"] if
                    key in ["T1_COR_agent_None", "T1_FS_COR_agent_GD", "T2_FS_COR_agent_None"]]
        ax_keys = [key for key in self.hparams["weightings"] if key in ["T1_FS_AX_agent_GD", "T2_FS_AX_agent_None"]]
        cor_meta_keys = ["T1_COR_agent_None_meta", "T1_FS_COR_agent_GD_meta", "T2_FS_COR_agent_None_meta"]
        cor_meta_keys_remaining = []
        ax_meta_keys = ["T1_FS_AX_agent_GD_meta", "T2_FS_AX_agent_None_meta"]
        ax_meta_keys_remaining = []
        for i, key in enumerate(["T1_COR_agent_None", "T1_FS_COR_agent_GD", "T2_FS_COR_agent_None"]):
            if key in self.hparams["weightings"]:
                cor_meta_keys_remaining.append(cor_meta_keys[i])
        for i, key in enumerate(["T1_FS_AX_agent_GD", "T2_FS_AX_agent_None"]):
            if key in self.hparams["weightings"]:
                ax_meta_keys_remaining.append(ax_meta_keys[i])
        all_sequence = ["T1_FS_AX_agent_GD", "T2_FS_AX_agent_None", "T1_COR_agent_None", "T1_FS_COR_agent_GD", "T2_FS_COR_agent_None"]
        sequences_to_drop = []
        for sequence in all_sequence:
            if sequence not in self.hparams["weightings"]:
                sequences_to_drop.append(sequence)
        pipeline = [
            {"ImageInput.DicomINd": {"keys": self.hparams["weightings"], "keep_metadata": True}},
            {"Transformation.Sitk2Numpy": {"keys": self.hparams["weightings"], "kwargs": None}},
            {"monai.transforms.CastToTyped": {"keys": self.hparams["weightings"], "dtype": np.float32}},
            {"monai.transforms.AddChanneld": {"keys": self.hparams["weightings"]}},
            {"ChangeDimensionOrder": {"keys": self.hparams["weightings"], "source_order": 1, "target_order": -1}},
            {"ValidCrop": {"keys": vc_first_keys, "border_percentage_x": 0.05}} if len(vc_first_keys) > 0 else None,
            {"ValidCrop": {"keys": vc_second_keys, "border_percentage_x": 0.05, "foreground_threshold": 0.1}} if len(
                vc_second_keys) > 0 else None,
            {"ValidCrop": {"keys": vc_third_keys, "border_percentage_x": 0.05, "foreground_threshold": 0.1}} if len(
                vc_third_keys) > 0 else None,

            # Original dimensions
            # T1_COR_agent_None: (500, 500, 17)
            # T1_FS_COR_agent_GD: (496, 496, 17)
            # T2_FS_COR_agent_None: (443, 443, 17)
            # T1_FS_AX_agent_GD: (314, 320, 65)
            # T2_FS_AX_agent_None: (334, 341, 65)

            {"monai.transforms.Resized": {"keys": cor_keys, "spatial_size": [512, 512, 16],
                                          "mode": "trilinear", "align_corners": False}} if len(cor_keys) > 0 else None,
            {"monai.transforms.Resized": {"keys": ax_keys, "spatial_size": [320, 320, 64],
                                          "mode": "trilinear", "align_corners": False}} if len(ax_keys) > 0 else None,
            {"ZScoreNormalization": {"keys": self.hparams["weightings"]}},
            {"ConditionalFlipd": {"keys": cor_keys, "meta_keys": cor_meta_keys_remaining, "axis": 0}} if len(
                cor_keys) > 0 else None,
            {"ConditionalFlipd": {"keys": ax_keys, "meta_keys": ax_meta_keys_remaining, "axis": 1}} if len(
                ax_keys) > 0 else None,
            {"DropKeyd": {"string": "meta"}},
            {"DropKeyd": {"string": sequences_to_drop}},
            {"NormalizeClinicalDatad": {}},
        ]
        pipeline = [transformation for transformation in pipeline if transformation is not None]
        return pipeline

    def train_dataloader(self) -> DataLoader:
        train_dataset = deepcopy(self.dataset)
        if self.hparams["CV"] is None:
            train_dataset.data = train_dataset.data[0:int(self.db_length * self.split[0])]
        else:
            number_of_items_per_fold = len(train_dataset.data) // 5
            train_dataset.data = [data for i, data in enumerate(train_dataset.data) if (i < self.hparams["CV"] * number_of_items_per_fold) or
                                                                                       (i >= (self.hparams["CV"] + 1) * number_of_items_per_fold)]
            train_dataset.data = train_dataset.data[0:int(len(train_dataset.data) * 0.8)]
            train_dataset.data.extend(self.additional_data_not_all_sequences)
        self.print_class_distribution_info(train_dataset.data, "Training")
        classes = [item["Class"] for item in train_dataset.data]
        return ArthritisDataLoader(train_dataset, shuffle=True, batch_size=self.batch_size,
                                   num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        val_dataset = deepcopy(self.dataset)
        if self.hparams["CV"] is None:
            val_dataset.data = val_dataset.data[int(self.db_length * self.split[0]):
                                                int(self.db_length * self.split[0] + self.db_length * self.split[1])]
        else:
            number_of_items_per_fold = len(val_dataset.data) // 5
            val_dataset.data = [data for i, data in enumerate(val_dataset.data) if (i < self.hparams["CV"] * number_of_items_per_fold) or
                                                                                   (i >= (self.hparams["CV"] + 1) * number_of_items_per_fold)]
            val_dataset.data = val_dataset.data[int(len(val_dataset.data) * 0.8):]
        self.print_class_distribution_info(val_dataset.data, "Validation")
        return ArthritisDataLoader(val_dataset, shuffle=False, batch_size=self.batch_size,
                                   num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        test_dataset = deepcopy(self.dataset)
        if self.hparams["CV"] is None:
            test_dataset.data = test_dataset.data[-int(self.db_length * self.split[2]):]
        else:
            number_of_items_per_fold = len(test_dataset.data) // 5
            test_dataset.data = [data for i, data in enumerate(test_dataset.data) if (i >= self.hparams["CV"] * number_of_items_per_fold) and
                                                                                     (i < (self.hparams["CV"] + 1) * number_of_items_per_fold)]
        self.print_class_distribution_info(test_dataset.data, "Test")
        return ArthritisDataLoader(test_dataset, shuffle=False, batch_size=self.batch_size,
                                   num_workers=self.num_workers)

    @staticmethod
    def print_class_distribution_info(data: list, phase: str):
        classes = [item["Class"] for item in data]
        for class_ in set(classes):
            print(f"INFO: {phase} Class distribution: Class {class_}: {classes.count(class_)}.")


class Model(pl.LightningModule):
    def __init__(self, hparams, phase="train"):
        super().__init__()
        self.save_hyperparameters()
        if phase == "train":
            self.model = get_pretrained_resnet(34, pretrained=True)
        elif phase == "test":
            self.model = Ensemble(self.hparams["hparams"]["indices"], self.hparams["hparams"]["CV"])
        self.clinical_keys = ["kerGewicht", "kerGroesse", "kerRaucherstatus",
                              "CRP",
                              "SJC",
                              "TJC",
                              "HAQ_Score",
                              "DAS_28",
                              "patGeschlecht"]
        self.custom_logger = []
        self.vis = GuidedGradCam(self.model, self.model.layer4[-1].conv2) 

    def forward(self, volume):
        return self.model(volume)

    def configure_optimizers(self):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams["hparams"]["lr"],
                                )
        scheduler = ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=0, verbose=True)
        optim_dict = {
        'optimizer': optim,
        'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'epoch',
            "monitor": "val/auc",
            }
        }
        return optim_dict

    def predict(self, input):
        return self(input)

    def training_step(self, batch, batch_idx):
        x = [batch[weighting] for weighting in self.hparams["hparams"]["weightings"]]
        y = batch["Class"]
        y_hat = self.predict(x)
        loss = F.cross_entropy(y_hat, y)
        with torch.no_grad():
            y_hat = F.softmax(y_hat, dim=1)
        return {'loss': loss, "train_y_hat": y_hat, "train_y": y}

    def validation_step(self, batch, batch_idx):
        x = [batch[weighting] for weighting in self.hparams["hparams"]["weightings"]]
        y = batch["Class"]
        y_hat = self.predict(x)
        loss = F.cross_entropy(y_hat, y)
        if self.current_epoch > 5:
            self.log_gradcam(x, y, batch_idx)
        with torch.no_grad():
            y_hat = F.softmax(y_hat, dim=1)
        return {'val_loss': loss, "val_y_hat": y_hat, "val_y": y}

    def test_step(self, batch, batch_idx):
        x = [batch[weighting] for weighting in self.hparams["hparams"]["weightings"]]
        y = batch["Class"]
        y_hat = self.predict(x)
        loss = F.cross_entropy(y_hat, y)
        with torch.no_grad():
            y_hat = F.softmax(y_hat, dim=1)
        return {'test_loss': loss, "test_y_hat": y_hat, "test_y": y}

    def log_gradcam(self, x, y, batch_idx=None):
        for b in range(x.shape[0]):
            label = y.detach().clone().squeeze()[b]
            input_gc = x.detach().clone()
            input_gc.requires_grad = True
            res = self.vis(x[b].unsqueeze(0))
            grad_cam_0 = res[0][..., 0].squeeze().cpu()
            grad_cam_1 = res[0][..., 1].squeeze().cpu()
            grad_cam_0 = (grad_cam_0 - grad_cam_0.min()) / (grad_cam_0.max() - grad_cam_0.min())
            grad_cam_0 = np.abs((grad_cam_0 - 0.5))
            grad_cam_0 = (grad_cam_0 - grad_cam_0.min()) / (grad_cam_0.max() - grad_cam_0.min())
            grad_cam_1 = (grad_cam_1 - grad_cam_1.min()) / (grad_cam_1.max() - grad_cam_1.min())
            grad_cam_1 = np.abs(grad_cam_1 - 0.5)
            grad_cam_1 = (grad_cam_1 - grad_cam_1.min()) / (grad_cam_1.max() - grad_cam_1.min())
            input_gc = (input_gc - input_gc.min()) / (input_gc.max() - input_gc.min())
            input_gc = input_gc[0].squeeze().cpu()
            for i in range(input_gc.shape[-1]):
                fig, ax = plt.subplots()
                ax.imshow(input_gc[..., i].squeeze(), cmap="gray", vmin=0, vmax=1)
                im = ax.imshow(grad_cam_1[..., i].squeeze(), cmap="jet", alpha=0.3, vmin=0, vmax=1)
                fig.colorbar(im, ax=ax)
                if batch_idx is not None:
                    if not os.path.exists(path_epoch := f"epoch_{self.current_epoch}"):
                        os.mkdir(path_epoch)
                    plt.savefig(os.path.join(path_epoch, f"case_{batch_idx}_batch_{b}_slice_{i}.jpg"))
                    plt.close(fig)
                else:
                    self.logger.experiment.add_figure(f"val/batch_{b}_occlusion_{i:02}_label_{label}", fig, global_step=self.current_epoch)

    def calculate_metrics_on_epoch_end(self, y, y_hat, avg_loss, phase: str):
        y = y.detach().cpu().numpy()
        y_hat_argmax =y_hat.argmax(axis=1)
        avg_accuracy = accuracy_score(y, y_hat_argmax)
        labels = [0, 1]

        class_report = classification_report(y, y_hat_argmax, output_dict=True, target_names=["class_0", "class_1"],
                                             zero_division=0, labels=labels)
        for key in class_report.keys():
            if "class_" in key:
                for metric in class_report[key].keys():
                    self.log(f"{phase}/{key}_{metric}", class_report[key][metric])
        avg_accuracy_classes = [accuracy_score(y == class_label, y_hat_argmax == class_label) for class_label in
                                labels]
        fig = plt.figure()
        aurocs = []
        for i in range(2):
            try:
                fpr, tpr, _ = roc_curve(y == i, y_hat[:, i])
                auroc = roc_auc_score(y == i, y_hat[:, i])
            except ValueError:
                auroc, fpr, tpr = 0.0, 0.0, 0.0
            plt.plot(fpr, tpr, label=f"Class {i} AUC {auroc * 100:.2f}")
            aurocs.append(auroc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend()
        self.log(f"{phase}/auc", np.mean(aurocs))
        self.logger.experiment.add_figure(f"{phase}/ROC", fig, global_step=self.current_epoch)

        if phase == "val":
            self.custom_logger.append(np.mean(aurocs))
        
        avg_precision = precision_score(y, y_hat_argmax, labels=labels, average='macro', zero_division=0)
        avg_recall = recall_score(y, y_hat_argmax, labels=labels, average='macro', zero_division=0)
        avg_f1 = f1_score(y, y_hat_argmax, labels=labels, average='macro', zero_division=0)
        for name, value in zip(
                [f"{phase}/loss", f"{phase}/accuracy",
                f"{phase}/precision", f"{phase}/recall",
                 f"{phase}/f1", *[f"{phase}/class_{c}_accuracy" for c in labels]],
                [avg_loss, avg_accuracy,
                avg_precision, avg_recall, avg_f1, *avg_accuracy_classes]):
            self.log(name, value)


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train/loss", avg_loss)
        y_hat = stack_tensors(outputs, "train_y_hat")
        y_hat = y_hat.detach().cpu().numpy()
        y = stack_tensors(outputs, "train_y")
        self.calculate_metrics_on_epoch_end(y, y_hat, avg_loss, "train")

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("val/loss", avg_loss)
        y_hat = stack_tensors(outputs, "val_y_hat")
        y_hat = y_hat.detach().cpu().numpy()
        y = stack_tensors(outputs, "val_y")
        self.calculate_metrics_on_epoch_end(y, y_hat, avg_loss, "val")

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log("test/loss", avg_loss)
        y_hat = stack_tensors(outputs, "test_y_hat")
        y_hat = y_hat.detach().cpu().numpy()
        y = stack_tensors(outputs, "test_y")
        self.calculate_metrics_on_epoch_end(y, y_hat, avg_loss, "test")

def calc_cv_results(result, indices):
    weightings = ["T1_COR_agent_None", "T1_FS_COR_agent_GD", "T2_FS_COR_agent_None", "T1_FS_AX_agent_GD", "T2_FS_AX_agent_None"]
    cv_result = {"auc": [], "acc": [], "f1": [], "sens": [], "spec": [], }
    for fold in ["0", "1", "2", "3", "4"]:
        cv_result["acc"].append(result[fold][0]["test/accuracy"])
        cv_result["auc"].append(result[fold][0]["test/auc"])
        cv_result["sens"].append(result[fold][0]["test/class_1_recall"])
        cv_result["spec"].append(result[fold][0]["test/class_0_recall"])
        cv_result["f1"].append(result[fold][0]["test/f1"])
    metrics = cv_result.keys()
    final_result = {}
    for key in metrics:
        final_result[key + "_mean"] = np.mean(cv_result[key])
        final_result[key + "std"] = np.std(cv_result[key])
    final_result["weightings"] = [weightings[w] for w in indices]
    with open(f'{str(indices)}_cv_result.json', 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)


def train():
    seed = 42
    for fold in [0, 1, 2, 3, 4]:
        for weightings in [
            ["T1_COR_agent_None"], ["T1_FS_COR_agent_GD"], ["T2_FS_COR_agent_None"],
            ["T1_FS_AX_agent_GD"], ["T2_FS_AX_agent_None"]
        ]:
            pl.seed_everything(seed)
            hparams = {"batch_size": 4,
                        "lr": 1e-4, "weightings": weightings,
                        "drop_prob": 0.2, "densenet_size": -1, "random_seed": seed,
                        "CV": fold}
            mri_dataset = DataModule(hparams)
            hparams["transformation_hparams"] = mri_dataset.transformation_hparams
            hparams["epochs"] = 9999
            model = Model(hparams, "train")
            trainer = pl.Trainer(gpus=[1], deterministic=True, default_root_dir=os.path.join(os.path.dirname(__file__), "CV_runs"),
                                max_epochs=hparams["epochs"], accelerator="ddp",
                                callbacks=[EarlyStopping(monitor="val/auc", patience=10, mode="max"),
                                            ModelCheckpoint(monitor="val/auc", mode="max")],
                                num_sanity_val_steps=0)
            trainer.fit(model, mri_dataset)

            shutil.rmtree(mri_dataset.cache_dir, ignore_errors=True)
            del trainer, mri_dataset, model
            torch.cuda.empty_cache()
            if "PL_EXP_VERSION" in list(os.environ.keys()):
                os.environ.pop("PL_EXP_VERSION")


def test():
        for indices in [
        # Select weightings to combine
        [0, 2, 4], [0, 2, 3], [0, 2, 3, 4], [0, 1], [0, 1, 4], [0, 1, 3], [0, 1, 3, 4], [0, 1, 2], [0, 1, 2, 4], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 3, 4], [0, 2],
        ]:
            result = {}
            for fold in [0, 1, 2, 3, 4]:
                weightings = ["T1_COR_agent_None", "T1_FS_COR_agent_GD", "T2_FS_COR_agent_None", "T1_FS_AX_agent_GD", "T2_FS_AX_agent_None"]
                print(f"Indices: {indices}, Fold: {fold}")
                weightings = [w for i, w in enumerate(weightings) if i in indices]
                pl.seed_everything(42)
                hparams = {"batch_size": 4,
                            "lr": 1e-4, "weightings": weightings,
                            "drop_prob": 0.2, "densenet_size": -1, "random_seed": 42,
                            "CV": fold, "indices": indices}
                mri_dataset = DataModule(hparams)
                hparams["transformation_hparams"] = mri_dataset.transformation_hparams
                hparams["epochs"] = 9999
                model = Model(hparams, "test")
                trainer = pl.Trainer(gpus=[1], deterministic=True, default_root_dir=os.path.join(os.path.dirname(__file__), "CV_runs"),
                                    max_epochs=hparams["epochs"], accelerator="ddp",
                                    callbacks=[EarlyStopping(monitor="val/auc", patience=10, mode="max"),
                                                ModelCheckpoint(monitor="val/auc", mode="max")],
                                    num_sanity_val_steps=0)
                mri_dataset.setup()
                result[str(fold)] = trainer.test(model, mri_dataset.test_dataloader())

                shutil.rmtree(mri_dataset.cache_dir, ignore_errors=True)
                del trainer, mri_dataset, model
                torch.cuda.empty_cache()
                if "PL_EXP_VERSION" in list(os.environ.keys()):
                    os.environ.pop("PL_EXP_VERSION")
            calc_cv_results(result, indices)


if __name__ == "__main__":
    train()
    test()

