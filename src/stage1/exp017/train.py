import argparse
import datetime
import math
import os
import warnings
from glob import glob
from typing import List

import albumentations as A
import cv2
import h5py
import numpy as np
import pandas as pd
import pydicom
import pytorch_lightning as pl
import timm
import torch
from albumentations.pytorch import ToTensorV2
from decoder_utils import FastFCNImproveHead, UNetHead
from pytorch_lightning import LightningDataModule, callbacks
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info
from sklearn.model_selection import GroupKFold
from timm.utils import ModelEmaV3
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

EXP_ID = "017"
COMMENT = """
keypoint detector, simcc like network, Simple 2DNet, ignore wrong label series, large rotate, dice_loss, use v2 volume, use emav3, fix dice ratio,
using clean keypoint, use all label, fix augmentations
"""


def get_transform(mode: str = "valid"):
    if mode == "train":
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.2,
                    rotate_limit=45,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.9,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(p=1),
                        A.GaussNoise(p=1),
                        A.PiecewiseAffine(p=1),  # IAAPiecewiseAffine
                    ],
                    p=0.3,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.3
                ),
                A.Resize(256, 256),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["class_labels"]
            ),
        )
    elif mode == "valid":
        transform = A.Compose(
            [ToTensorV2()],
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["class_labels"]
            ),
        )
    else:
        transform = A.Compose(
            [ToTensorV2()],
        )
    return transform


class RSNA2024Stage1Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str = "train",  # "train"  | "valid" | "test"
    ):
        self.mode = mode
        self.train = mode == "train"
        self.df = df
        self.series_ids = self.df.series_id.unique()
        self.transform = get_transform(self.train)

    def __len__(self) -> int:
        return len(self.series_ids)

    def __getitem__(self, idx: int):
        series_df = self.df[self.df.series_id == self.series_ids[idx]]
        h5f = h5py.File("../../../input/resize_volume_256x256_v2.h5")
        volume = h5f[str(series_df.series_id.iloc[0])][:]  # (20, 256, 256)
        if len(series_df) == 10:
            new_series_df = series_df.iloc[::2].reset_index(drop=True).copy()
            new_series_df_l = series_df.iloc[::2].reset_index(drop=True).copy()
            new_series_df_r = series_df.iloc[1::2].reset_index(drop=True).copy()
            new_series_df = new_series_df.astype(
                {"relative_x": np.float32, "relative_y": np.float32}
            )
            new_series_df.loc[:, ["relative_x", "relative_y"]] = (
                new_series_df_l.loc[:, ["relative_x", "relative_y"]].to_numpy()
                + new_series_df_r.loc[:, ["relative_x", "relative_y"]].to_numpy()
            ) / 2
            new_series_df.loc[:, "instance_number_l"] = new_series_df_l.loc[
                :, "instance_number"
            ].to_numpy()
            new_series_df.loc[:, "instance_number_r"] = new_series_df_r.loc[
                :, "instance_number"
            ].to_numpy()
        else:
            assert len(series_df) == 5
            new_series_df = series_df.reset_index(drop=True)
        img_paths = glob(
            f"../../../input/rsna-2024-lumbar-spine-degenerative-classification/train_images/**/{series_df.series_id.iloc[0]}/*.dcm"
        )
        keypoints = np.zeros((5, 2), dtype=int)
        for row_id in range(len(new_series_df)):
            row = new_series_df.iloc[row_id]
            keypoints[row_id, 0] = int(row.relative_x * volume.shape[2])
            keypoints[row_id, 1] = int(row.relative_y * volume.shape[1])
        class_labels = [0, 1, 2, 3, 4]
        if self.mode == "test":
            transformed = self.transform(
                image=volume.transpose((1, 2, 0)),
            )
            volume = transformed["image"] / 255.0  # (c, x, y)
            return {
                "volume": volume,
                "series_id": img_paths[0].split("/")[-2],
                "study_id": img_paths[0].split("/")[-3],
            }
        else:
            transformed = self.transform(
                image=volume.transpose((1, 2, 0)),
                keypoints=keypoints,
                class_labels=class_labels,
            )
            keypoints = np.asarray(
                transformed["keypoints"],
            )
            label = np.zeros((5, volume.shape[1], volume.shape[2]))
            class_id_map = {i: c for i, c in enumerate(transformed["class_labels"])}
            for row_id in range(len(new_series_df)):
                if row_id in transformed["class_labels"]:
                    x_pos = keypoints[class_id_map[row_id], 0]
                    y_pos = keypoints[class_id_map[row_id], 1]
                    label_tmp = cv2.circle(
                        np.zeros((volume.shape[1], volume.shape[2], 3), dtype=np.uint8),
                        (int(x_pos), int(y_pos)),
                        7,
                        (1, 1, 1),
                        -1,
                    )[..., 0]
                    label[row_id] = label_tmp.astype(np.float32)

            label = torch.tensor(label)  # (ch, x, y)
            volume = transformed["image"] / 255.0  # (c, x, y)
            return {
                "volume": volume,
                "label": label,
                "series_id": img_paths[0].split("/")[-2],
                "study_id": img_paths[0].split("/")[-3],
            }


class RSNA2024Stage1DataModule(LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        num_workers: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.save_hyperparameters(ignore=["train_df", "valid_df"])

    def create_dataset(self, mode: str = "train") -> RSNA2024Stage1Dataset:
        if mode == "train":
            return RSNA2024Stage1Dataset(
                df=self.train_df,
                mode="train",
            )
        else:
            return RSNA2024Stage1Dataset(
                df=self.valid_df,
                mode="valid",
            )

    def __dataloader(self, mode: str = "train") -> DataLoader:
        """Train/validation loaders."""
        dataset = self.create_dataset(mode)
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=(mode == "train"),
            drop_last=(mode == "train"),
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="train")

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="valid")

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="test")

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("RSNA2024Stage1DataModule")
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            metavar="W",
            help="number of CPU workers",
            dest="num_workers",
        )
        parser.add_argument(
            "--batch_size",
            default=16,
            type=int,
            metavar="BS",
            help="number of sample in a batch",
            dest="batch_size",
        )
        return parent_parser


class RSNA2024Stage1Model(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_path_rate: float = 0,
        in_chans: int = 20,
        num_class: int = 5,
        img_size: List[int] = 256,
        decoder_type: str = "UNet",  # UNet or FastFCNImprove
        center: str = None,
        attention_type: str = None,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            drop_path_rate=drop_path_rate,
            img_size=(
                img_size
                if ("swin" in model_name)
                or ("coat" in model_name)
                or ("max" in model_name)
                else None
            ),
        )
        self.model_name = model_name
        self.output_fmt = getattr(self.encoder, "output_fmt", "NHCW")
        num_features = self.encoder.feature_info.channels()
        if decoder_type == "UNet":
            self.head = UNetHead(
                encoder_channels=num_features,
                num_class=num_class,
                center=center,
                attention_type=attention_type,
                classification=False,
                deep_supervision=False,
            )
        elif decoder_type == "FastFCNImprove":
            self.head = FastFCNImproveHead(
                encoder_channels=num_features,
                num_class=num_class,
                attention_type=attention_type,
                classification=False,
                deep_supervision=False,
            )
        else:
            raise NotImplementedError

    def forward(
        self,
        volume: torch.Tensor,
    ):
        """
        img: (bs, z * ch, h, w)
        """
        bs, ch, h, w = volume.shape
        assert ch == self.in_chans
        volume_feats = self.encoder(volume)
        if self.output_fmt == "NHWC":
            volume_feats = [
                feat.permute(0, 3, 1, 2).contiguous() for feat in volume_feats
            ]
        out = self.head(volume_feats)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return {"logit": out}

    def set_grad_checkpointing(self, enable: bool = True):
        self.encoder.set_grad_checkpointing(enable)


class Mixup(object):
    def __init__(self, p: float = 0.5, alpha: float = 0.5):
        self.p = p
        self.alpha = alpha
        self.lam = 1.0
        self.do_mixup = False

    def init_lambda(self):
        if np.random.rand() < self.p:
            self.do_mixup = True
        else:
            self.do_mixup = False
        if self.do_mixup and self.alpha > 0.0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0

    def reset_lambda(self):
        self.lam = 1.0


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        y_true_count = targets.sum((1, 2, 3))  # (bs)
        ctp = (preds * targets).sum((1, 2, 3))  # (bs)
        cfp = (preds * (1 - targets)).sum((1, 2, 3))  # (bs)

        c_precision = ctp / (ctp + cfp + self.smooth)
        c_recall = ctp / (y_true_count + self.smooth)
        dice = 2 * (c_precision * c_recall) / (c_precision + c_recall + self.smooth)
        return 1 - dice.mean()


class RSNA2024Stage1LightningModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_path_rate: float = 0,
        decoder_type: str = "UNet",  # UNet or FastFCNImprove
        center: str = None,
        attention_type: str = None,
        in_chans: int = 20,
        num_class: int = 5,
        mixup_p: float = 0.0,
        mixup_alpha: float = 0.5,
        no_mixup_epochs: int = 0,
        lr: float = 1e-3,
        backbone_lr: float = None,
        weight_decay: float = 0.0001,
        enable_gradient_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.__build_model(
            model_name=model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            decoder_type=decoder_type,
            center=center,
            attention_type=attention_type,
            in_chans=in_chans,
            num_class=num_class,
        )
        if enable_gradient_checkpoint:
            self.model.set_grad_checkpointing(enable_gradient_checkpoint)
        self.mixupper = Mixup(p=mixup_p, alpha=mixup_alpha)
        self.hparams.backbone_lr = (
            self.hparams.backbone_lr if self.hparams.backbone_lr is not None else lr
        )

    def __build_model(
        self,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_path_rate: float = 0,
        decoder_type: str = "UNet",  # UNet or FastFCNImprove
        center: str = None,
        attention_type: str = None,
        in_chans: int = 20,
        num_class: int = 5,
    ):
        self.model = RSNA2024Stage1Model(
            model_name=model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            decoder_type=decoder_type,
            center=center,
            attention_type=attention_type,
            in_chans=in_chans,
            num_class=num_class,
        )
        self.model_ema = ModelEmaV3(
            self.model,
            decay=0.99,
        )
        self.criterions = {
            "bce": nn.BCEWithLogitsLoss(),
            "dice": DiceLoss(),
        }

    def calc_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        losses = {}
        # smooth = 0.1
        # true = labels["targets"] * (1 - (smooth / 0.5)) + smooth

        # TODO: implement loss calc
        losses["bce"] = self.criterions["bce"](outputs["logit"], labels["label"])
        losses["dice"] = self.criterions["dice"](outputs["logit"], labels["label"])
        losses["loss"] = losses["bce"] * 0.2 + losses["dice"] * 0.8
        return losses

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model)
        step_output = {}
        outputs = {}
        loss_target = {}
        self.mixupper.init_lambda()
        volume, label = (
            batch["volume"],
            batch["label"],
        )
        if (
            self.mixupper.do_mixup
            and self.current_epoch
            < self.trainer.max_epochs - self.hparams.no_mixup_epochs
        ):
            volume = self.mixupper.lam * volume + (1 - self.mixupper.lam) * volume.flip(
                0
            )
        outputs.update(self.model(volume))
        loss_target["label"] = label
        losses = self.calc_loss(outputs, loss_target)
        if (
            self.mixupper.do_mixup
            and self.current_epoch
            < self.trainer.max_epochs - self.hparams.no_mixup_epochs
        ):
            loss_target["label"] = label.flip(0)
            losses_b = self.calc_loss(outputs, loss_target)
            for key in losses:
                losses[key] = (
                    self.mixupper.lam * losses[key]
                    + (1 - self.mixupper.lam) * losses_b[key]
                )
        step_output.update(losses)

        self.log_dict(
            dict(
                train_loss=losses["loss"],
                train_bce_loss=losses["bce"],
                train_dice_loss=losses["dice"],
            ),
            sync_dist=True,
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = {}
        outputs = {}
        loss_target = {}

        volume, label = (
            batch["volume"],
            batch["label"],
        )
        outputs.update(self.model_ema.module(volume))
        loss_target["label"] = label
        losses = self.calc_loss(outputs, loss_target)

        step_output.update(losses)

        self.log_dict(
            dict(
                val_loss=losses["loss"],
                val_bce_loss=losses["bce"],
                val_dice_loss=losses["dice"],
            ),
            sync_dist=True,
        )
        return step_output

    def on_validation_epoch_end(self):
        pass

    def get_optimizer_parameters(self):
        no_decay = ["bias", "gamma", "beta"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.hparams.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.head.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.hparams.lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.head.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.lr,
            },
        ]
        return optimizer_parameters

    def configure_optimizers(self):
        self.warmup = True
        eps = 1e-8 if self.trainer.precision == "32-true" else 1e-6
        optimizer = AdamW(self.get_optimizer_parameters(), eps=eps)
        max_train_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil(max_train_steps / 50) if self.warmup else 0
        rank_zero_info(
            f"max_train_steps: {max_train_steps}, warmup_steps: {warmup_steps}"
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("RSNA2024Stage1LightningModel")
        parser.add_argument(
            "--model_name",
            default="resnet34",
            type=str,
            metavar="MN",
            help="Name (as in ``timm``) of the feature extractor",
            dest="model_name",
        )
        parser.add_argument(
            "--drop_path_rate",
            default=None,
            type=float,
            metavar="DPR",
            dest="drop_path_rate",
        )
        parser.add_argument(
            "--decoder_type",
            default="UNet",
            type=str,
            choices=["UNet", "FastFCNImprove"],
            metavar="DT",
            help="Name of the decoder_type, implemented: UNet|FastFCNImproved",
            dest="decoder_type",
        )
        parser.add_argument(
            "--center",
            default=None,
            type=str,
            choices=[None, "fpa", "aspp"],
            metavar="CT",
            help="Name of the center module, implemented: None|fpa|aspp",
            dest="center",
        )
        parser.add_argument(
            "--attention_type",
            default=None,
            type=str,
            choices=[None, "scse", "cbam"],
            metavar="AT",
            help="Name of the attention module, implemented: None|scse|cbam",
            dest="attention_type",
        )
        parser.add_argument(
            "--in_chans",
            default=20,
            type=int,
            metavar="ICH",
            dest="in_chans",
        )
        parser.add_argument(
            "--num_class",
            default=5,
            type=int,
            metavar="OCL",
            dest="num_class",
        )
        parser.add_argument(
            "--mixup_p", default=0.0, type=float, metavar="MP", dest="mixup_p"
        )
        parser.add_argument(
            "--mixup_alpha", default=0.0, type=float, metavar="MA", dest="mixup_alpha"
        )
        parser.add_argument(
            "--no_mixup_epochs",
            default=0,
            type=int,
            metavar="NME",
            dest="no_mixup_epochs",
        )
        parser.add_argument(
            "--lr",
            default=1e-3,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--backbone_lr",
            default=None,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="backbone_lr",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0001,
            type=float,
            metavar="WD",
            help="initial weight decay",
            dest="weight_decay",
        )
        parent_parser.add_argument(
            "--enable_gradient_checkpoint",
            action="store_true",
            help="enable set_gradient_checkpoint",
            dest="enable_gradient_checkpoint",
        )
        return parent_parser


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=True)
    parent_parser.add_argument(
        "--seed",
        default=2024,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="1 batch run for debug",
        dest="debug",
    )
    # dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"tmp",
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parent_parser.add_argument(
        "--gpus", type=int, default=4, help="number of gpus to use"
    )
    parent_parser.add_argument(
        "--epochs", default=10, type=int, metavar="N", help="total number of epochs"
    )
    parent_parser.add_argument(
        "--precision",
        # default="16-mixed",
        default="32",
    )
    parent_parser.add_argument(
        "--series_description", default="Sagittal_T1"
    )  # "Sagittal_T2/STIR"
    parser = RSNA2024Stage1LightningModel.add_model_specific_args(parent_parser)
    parser = RSNA2024Stage1DataModule.add_model_specific_args(parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed, workers=True)
    if not args.debug:
        warnings.simplefilter("ignore")
    assert args.fold < 5
    train_descriptions = pd.read_csv(
        "../../../input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv"
    )
    clean_keypoints = pd.read_csv(
        "../../../input/lumbar-coordinate-pretraining-dataset/coords_rsna_improved.csv",
        index_col=0,
    )
    train_coord_df = pd.concat(
        [
            clean_keypoints[
                (clean_keypoints.condition == "Spinal Canal Stenosis")
                & (clean_keypoints.side == "R")
            ],
            clean_keypoints[~(clean_keypoints.condition == "Spinal Canal Stenosis")],
        ]
    )
    train_coord_df = train_coord_df.merge(
        train_descriptions, on=["series_id", "study_id"], how="left"
    )
    train_df = pd.read_csv(
        "../../../input/rsna-2024-lumbar-spine-degenerative-classification/train.csv"
    )
    train_df["fold_id"] = -1
    for i, (train_index, valid_index) in enumerate(
        GroupKFold(n_splits=5).split(
            train_df, np.arange(len(train_df)), train_df.study_id
        )
    ):
        train_df.loc[valid_index, "fold_id"] = i
    train_coord_df = train_coord_df.merge(
        train_df.loc[:, ["study_id", "fold_id"]], on=["study_id"], how="left"
    )
    new_train_coord_df = train_coord_df.sort_values(
        by=["study_id", "series_id", "level", "condition"]
    ).reset_index(drop=True)
    wrong_series_ids = [3951475160]
    new_train_coord_df = new_train_coord_df[
        ~new_train_coord_df.series_id.isin(wrong_series_ids)
    ]

    new_train_coord_df = new_train_coord_df[
        new_train_coord_df.series_description
        == args.series_description.replace("_", " ").replace("-", "/")
    ].reset_index()

    for fold in range(5):
        if fold != args.fold:
            continue
        train_df = new_train_coord_df[new_train_coord_df.fold_id != fold].reset_index(
            drop=True
        )
        valid_df = new_train_coord_df[new_train_coord_df.fold_id == fold].reset_index(
            drop=True
        )
        datamodule = RSNA2024Stage1DataModule(
            train_df=train_df,
            valid_df=valid_df,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        model = RSNA2024Stage1LightningModel(
            model_name=args.model_name,
            pretrained=True,
            drop_path_rate=args.drop_path_rate,
            decoder_type=args.decoder_type,
            center=args.center,
            attention_type=args.attention_type,
            in_chans=args.in_chans,
            num_class=args.num_class,
            mixup_p=args.mixup_p,
            mixup_alpha=args.mixup_alpha,
            no_mixup_epochs=args.no_mixup_epochs,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
            enable_gradient_checkpoint=args.enable_gradient_checkpoint,
        )

        logdir = f"../../../logs/stage1/exp{EXP_ID}/{args.logdir}/{args.series_description}/fold{fold}"
        print(f"logdir = {logdir}")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
            mode="min",
        )
        os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
        if not args.debug:
            wandb_logger = WandbLogger(
                name=f"stage1/exp{EXP_ID}/{args.logdir}/{args.series_description}/fold{fold}",
                tags=[f"fold{fold}", "stage1", args.series_description],
                save_dir=logdir,
                project="rsna-2024-lumbar-spine-degenerative-classification",
                notes=COMMENT,
            )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, log_rank_zero_only=True
        )
        trainer = pl.Trainer(
            default_root_dir=logdir,
            sync_batchnorm=True,
            gradient_clip_val=1.0,
            precision=args.precision,
            devices=args.gpus,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            # strategy="ddp",
            max_epochs=args.epochs,
            logger=wandb_logger if not args.debug else True,
            callbacks=[
                loss_checkpoint,
                lr_monitor,
                early_stopping,
            ],
            fast_dev_run=args.debug,
            num_sanity_val_steps=0,
            accumulate_grad_batches=max(16 // args.batch_size, 1),
        )
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main(get_args())
