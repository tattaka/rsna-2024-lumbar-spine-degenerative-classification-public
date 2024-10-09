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
import sklearn
import timm
import torch
from albumentations.pytorch import ToTensorV2
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

EXP_ID = "107"
COMMENT = """
stage2 classification, baseline, transformer, add layernorm, attention pooling, 
add 1d cnn, concat sagittal_t1 and Sagittal_t2-STIR, common extractor, token mask augmentation,
EMA, simple avg+max feat only for axial, add transformer option, refactor, simplied head, add pe
add mask, add max pooling, aux head, ax lr images, add random masking option, w/o randomresizecrop,
split sagittal crop_range, w/ competition metric, exp054 refactoring, 
using axial prediciton (clean), using sagittal prediciton(clean), fix crop scale, heavy augmentations,
fix preprocess(resize -> resize and pad)
"""
label_map = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}


def get_transform(mode: str = "valid", img_size: int = 128):
    if mode == "train":
        transform = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.25,
                    rotate_limit=45,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.9,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(),
                        A.GridDistortion(),
                        A.ElasticTransform(),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(),
                        A.MotionBlur(),
                        A.MedianBlur(),
                    ],
                    p=0.3,
                ),
                A.PiecewiseAffine(p=0.2),  # IAAPiecewiseAffine
                A.CoarseDropout(
                    max_height=int(img_size * 0.2),
                    max_width=int(img_size * 0.2),
                    max_holes=5,
                    p=0.5,
                ),
                ToTensorV2(),
            ],
        )
    elif mode == "valid":
        transform = A.Compose(
            [A.Resize(img_size, img_size), ToTensorV2()],
        )
    else:
        transform = A.Compose(
            [A.Resize(img_size, img_size), ToTensorV2()],
        )
    return transform


def dist(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_sorted_instance_number(series_id, series_description="Axial_T2"):
    if series_description == "Axial_T2":
        planes = 2
    else:
        planes = 0
    dicom_paths = glob(
        f"../../../input/rsna-2024-lumbar-spine-degenerative-classification/**/{series_id}/*.dcm",
        recursive=True,
    )
    positions = np.asarray(
        [
            float(pydicom.dcmread(dicom_path).ImagePositionPatient[planes])
            for dicom_path in dicom_paths
        ]
    )
    idx = np.argsort(-positions)
    return np.asarray([int(p.split("/")[-1].split(".")[0]) for p in dicom_paths])[idx]


def crop_axial(
    volume,
    img_size,
    in_chans,
    keypoint,
    instance_number,
    d,
    transform,
):
    cropped_volume = volume[
        max(int(instance_number) - in_chans // 2, 0) : min(
            int(instance_number) + int(np.ceil(in_chans / 2)),
            volume.shape[0],
        ),
        int(max(keypoint[1] - d, 0) * volume.shape[1]) : int(
            min(keypoint[1] + d, 1) * volume.shape[1]
        ),
        int(max(keypoint[0] - d, 0) * volume.shape[2]) : int(
            min(keypoint[0] + d, 1) * volume.shape[2]
        ),
    ]
    cropped_volume = cropped_volume.astype(np.float32)
    transformed = transform(
        image=cropped_volume.transpose((1, 2, 0)),
    )
    cropped_volume = transformed["image"]  # (c, x, y)
    pad_mask = torch.zeros(in_chans)
    if cropped_volume.shape[0] > in_chans:
        cropped_volume = F.interpolate(
            cropped_volume[None, None, ...],
            size=(in_chans, img_size, img_size),
            mode="trilinear",
            align_corners=True,
        )[0, 0]
    else:
        pad = in_chans - cropped_volume.shape[0]
        cropped_volume = F.pad(cropped_volume, (0, 0, 0, 0, 0, pad))
        pad_mask[-pad:] = 1
    return cropped_volume, pad_mask


def crop_sagittal(volume, img_size, in_chans, keypoint, d, transform):
    cropped_volume = volume[
        :,
        int(max(keypoint[1] - d, 0) * volume.shape[1]) : int(
            min(keypoint[1] + d, 1) * volume.shape[1]
        ),
        int(max(keypoint[0] - d, 0) * volume.shape[2]) : int(
            min(keypoint[0] + d, 1) * volume.shape[2]
        ),
    ]
    cropped_volume = cropped_volume.astype(np.float32)
    transformed = transform(
        image=cropped_volume.transpose((1, 2, 0)),
    )
    cropped_volume = transformed["image"]  # (c, x, y)
    pad_mask = torch.zeros(in_chans)
    if cropped_volume.shape[0] > in_chans:
        cropped_volume = F.interpolate(
            cropped_volume[None, None, ...],
            size=(in_chans, img_size, img_size),
            mode="trilinear",
            align_corners=True,
        )[0, 0]
    else:
        pad = in_chans - cropped_volume.shape[0]
        cropped_volume = F.pad(cropped_volume, (0, 0, 0, 0, 0, pad))
        pad_mask[-pad:] = 1
    return cropped_volume, pad_mask


class RSNA2024Stage2Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str = "train",  # "train"  | "valid" | "test"
        img_size_st1: int = 128,
        in_chans_st1: int = 20,
        crop_range_st1: float = 1,
        img_size_st2: int = 128,
        in_chans_st2: int = 20,
        crop_range_st2: float = 1,
        img_size_ax: int = 256,
        in_chans_ax: int = 3,
        crop_range_ax: float = 2,
    ):
        self.mode = mode
        self.train = mode == "train"
        self.df = df
        self.img_size_st1 = img_size_st1
        self.in_chans_st1 = in_chans_st1
        self.crop_range_st1 = crop_range_st1

        self.img_size_st2 = img_size_st2
        self.in_chans_st2 = in_chans_st2
        self.crop_range_st2 = crop_range_st2

        self.img_size_ax = img_size_ax
        self.in_chans_ax = in_chans_ax
        self.crop_range_ax = crop_range_ax

        self.transform_st1 = get_transform(mode, self.img_size_st1)
        self.transform_st2 = get_transform(mode, self.img_size_st2)
        self.transform_ax = get_transform(mode, self.img_size_ax)
        self.h5f = h5py.File("../../../input/volume_orig_res.h5")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        study_id = self.df.study_id.iloc[0]
        series_id_st1 = self.df.series_id_st1.iloc[idx]
        series_id_st2 = self.df.series_id_st2.iloc[idx]
        series_id_ax = self.df.series_id_ax.iloc[idx]
        level = self.df.level.iloc[idx]

        label = [
            label_map[self.df.left_neural_foraminal_narrowing.iloc[idx]],
            label_map[self.df.right_neural_foraminal_narrowing.iloc[idx]],
            label_map[self.df.spinal_canal_stenosis.iloc[idx]],
            label_map[self.df.left_subarticular_stenosis.iloc[idx]],
            label_map[self.df.right_subarticular_stenosis.iloc[idx]],
        ]

        # st1
        series_df = (
            self.df[self.df.series_id_st1 == series_id_st1]
            .groupby("level")
            .head(1)
            .sort_values("level")
            .reset_index(drop=True)
        )
        new_idx = np.arange(len(series_df))[series_df.level == level][0]
        volume = self.h5f[str(series_id_st1)]
        dists = []
        if new_idx + 1 < len(series_df):
            dists.append(
                dist(
                    series_df.iloc[new_idx].x_st1,
                    series_df.iloc[new_idx + 1].x_st1,
                    series_df.iloc[new_idx].y_st1,
                    series_df.iloc[new_idx + 1].y_st1,
                )
            )
        if new_idx - 1 >= 0:
            dists.append(
                dist(
                    series_df.iloc[new_idx].x_st1,
                    series_df.iloc[new_idx - 1].x_st1,
                    series_df.iloc[new_idx].y_st1,
                    series_df.iloc[new_idx - 1].y_st1,
                )
            )
        d = sum(dists) / len(dists) * self.crop_range_st1
        keypoint = np.asarray(
            [series_df.iloc[new_idx].x_st1, series_df.iloc[new_idx].y_st1]
        )
        cropped_volume_st1, pad_mask_st1 = crop_sagittal(
            volume,
            self.img_size_st1,
            self.in_chans_st1,
            keypoint,
            d,
            self.transform_st1,
        )

        series_df = (
            self.df[self.df.series_id_st2 == series_id_st2]
            .groupby("level")
            .head(1)
            .sort_values("level")
            .reset_index(drop=True)
        )
        new_idx = np.arange(len(series_df))[series_df.level == level][0]
        volume = self.h5f[str(series_id_st2)]
        dists = []
        if new_idx + 1 < len(series_df):
            dists.append(
                dist(
                    series_df.iloc[new_idx].x_st2,
                    series_df.iloc[new_idx + 1].x_st2,
                    series_df.iloc[new_idx].y_st2,
                    series_df.iloc[new_idx + 1].y_st2,
                )
            )
        if new_idx - 1 >= 0:
            dists.append(
                dist(
                    series_df.iloc[new_idx].x_st2,
                    series_df.iloc[new_idx - 1].x_st2,
                    series_df.iloc[new_idx].y_st2,
                    series_df.iloc[new_idx - 1].y_st2,
                )
            )
        d = sum(dists) / len(dists) * self.crop_range_st2
        keypoint = np.asarray(
            [series_df.iloc[new_idx].x_st2, series_df.iloc[new_idx].y_st2]
        )
        cropped_volume_st2, pad_mask_st2 = crop_sagittal(
            volume,
            self.img_size_st2,
            self.in_chans_st2,
            keypoint,
            d,
            self.transform_st2,
        )

        # axial t2
        volume_ax = self.h5f[str(series_id_ax)]
        series_df = (
            self.df[self.df.series_id_ax == series_id_ax]
            .groupby("level")
            .head(1)
            .sort_values("level")
            .reset_index(drop=True)
        )
        new_idx = np.arange(len(series_df))[series_df.level == level][0]
        x_l_ax = series_df.x_l_ax.iloc[new_idx]
        x_r_ax = series_df.x_r_ax.iloc[new_idx]
        y_l_ax = series_df.y_l_ax.iloc[new_idx]
        y_r_ax = series_df.y_r_ax.iloc[new_idx]
        keypoint_l = np.asarray([x_l_ax, y_l_ax])
        keypoint_r = np.asarray([x_r_ax, y_r_ax])
        instance_number_ax = int(
            series_df.instance_number_ax.iloc[new_idx] * volume_ax.shape[0]
        )

        d_ax = dist(x_l_ax, x_r_ax, y_l_ax, y_r_ax) * self.crop_range_ax
        cropped_volume_ax_l, pad_mask_ax_l = crop_axial(
            volume_ax,
            self.img_size_ax,
            self.in_chans_ax,
            keypoint_l,
            instance_number_ax,
            d_ax,
            self.transform_ax,
        )
        cropped_volume_ax_r, pad_mask_ax_r = crop_axial(
            volume_ax,
            self.img_size_ax,
            self.in_chans_ax,
            keypoint_r,
            instance_number_ax,
            d_ax,
            self.transform_ax,
        )
        cropped_volume_ax = torch.cat([cropped_volume_ax_l, cropped_volume_ax_r], 0)
        pad_mask_ax = torch.cat([pad_mask_ax_l, pad_mask_ax_r])

        if self.mode == "test":
            return {
                "volume_st1": cropped_volume_st1.float(),
                "volume_st2": cropped_volume_st2.float(),
                "volume_ax": cropped_volume_ax.float(),
                "pad_mask_st1": pad_mask_st1.float(),
                "pad_mask_st2": pad_mask_st2.float(),
                "pad_mask_ax": pad_mask_ax.float(),
                "series_id_st1": series_id_st1,
                "series_id_st2": series_id_st2,
                "series_id_ax": series_id_ax,
                "study_id": study_id,
                "level": level,
            }
        else:
            label = torch.tensor(label)  # (ch, x, y)
            return {
                "volume_st1": cropped_volume_st1.float(),
                "volume_st2": cropped_volume_st2.float(),
                "volume_ax": cropped_volume_ax.float(),
                "pad_mask_st1": pad_mask_st1.float(),
                "pad_mask_st2": pad_mask_st2.float(),
                "pad_mask_ax": pad_mask_ax.float(),
                "series_id_st1": series_id_st1,
                "series_id_st2": series_id_st2,
                "series_id_ax": series_id_ax,
                "label": label,
                "study_id": study_id,
                "level": level,
            }


class RSNA2024Stage2DataModule(LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        img_size_st1: int = 128,
        in_chans_st1: int = 20,
        crop_range_st1: float = 1,
        img_size_st2: int = 128,
        in_chans_st2: int = 20,
        crop_range_st2: float = 1,
        img_size_ax: int = 128,
        in_chans_ax: int = 5,
        crop_range_ax: float = 2,
        num_workers: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.save_hyperparameters(ignore=["train_df", "valid_df"])

    def create_dataset(self, mode: str = "train") -> RSNA2024Stage2Dataset:
        if mode == "train":
            return RSNA2024Stage2Dataset(
                df=self.train_df,
                mode="train",
                img_size_st1=self.hparams.img_size_st1,
                in_chans_st1=self.hparams.in_chans_st1,
                crop_range_st1=self.hparams.crop_range_st1,
                img_size_st2=self.hparams.img_size_st2,
                in_chans_st2=self.hparams.in_chans_st2,
                crop_range_st2=self.hparams.crop_range_st2,
                img_size_ax=self.hparams.img_size_ax,
                in_chans_ax=self.hparams.in_chans_ax,
                crop_range_ax=self.hparams.crop_range_ax,
            )
        else:
            return RSNA2024Stage2Dataset(
                df=self.valid_df,
                mode="valid",
                img_size_st1=self.hparams.img_size_st1,
                in_chans_st1=self.hparams.in_chans_st1,
                crop_range_st1=self.hparams.crop_range_st1,
                img_size_st2=self.hparams.img_size_st2,
                in_chans_st2=self.hparams.in_chans_st2,
                crop_range_st2=self.hparams.crop_range_st2,
                img_size_ax=self.hparams.img_size_ax,
                in_chans_ax=self.hparams.in_chans_ax,
                crop_range_ax=self.hparams.crop_range_ax,
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
        parser = parent_parser.add_argument_group("RSNA2024Stage2DataModule")
        parser.add_argument(
            "--crop_range_st1",
            default=1.0,
            type=float,
            metavar="CRST1",
            help="range for keypoint cropping",
            dest="crop_range_st1",
        )
        parser.add_argument(
            "--crop_range_st2",
            default=1.0,
            type=float,
            metavar="CRST2",
            help="range for keypoint cropping",
            dest="crop_range_st2",
        )
        parser.add_argument(
            "--crop_range_ax",
            default=1,
            type=float,
            metavar="CRA",
            help="axial t2 range for keypoint cropping",
            dest="crop_range_ax",
        )
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


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    """
    Simple class for flattening layer.
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(
            input_dim, input_dim
        )  # 重みを計算するための線形層

    def forward(self, input):
        """
        x: (batch_size, seq_length, input_dim) のテンソル
        """
        if isinstance(input, dict):
            x, mask = input["x"], input["mask"]
        else:
            x = input
            mask = None
        # Attention スコアの計算 (batch_size, seq_length, input_dim)
        attention_scores = self.attention_weights(x)

        # ソフトマックス関数でスコアを正規化 (batch_size, seq_length, input_dim)
        if mask is not None:
            attention_scores[mask] = -float("inf")
        attention_weights = F.softmax(attention_scores, dim=1)

        # # 重み付け平均を計算してプーリング (batch_size, input_dim)
        weighted_sum = torch.sum(attention_weights * x, dim=1)

        return weighted_sum


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


# https://github.com/DrHB/icecube-2nd-place/blob/main/src/models.py
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int = 16, M: int = 10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RSNA2024Stage2Model(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        num_class: int = 3,
        model_name: str = "resnet34",
        drop_path_rate: float = 0,
        img_size_st1: int = 128,
        img_size_st2: int = 128,
        img_size_ax: int = 128,
        transformer_dim: int = 256,
        transformer_num_layers: int = 2,
        transformer_nhead: int = 8,
        max_token_mask_rate: float = 0.0,
    ):
        super().__init__()
        self.max_token_mask_rate = max_token_mask_rate
        if ("swin" in model_name) or ("coat" in model_name) or ("max" in model_name):
            assert img_size_st1 == img_size_st2
            assert img_size_st1 == img_size_ax

        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            features_only=True,
            drop_path_rate=drop_path_rate,
            img_size=(
                img_size_st1
                if ("swin" in model_name)
                or ("coat" in model_name)
                or ("max" in model_name)
                or ("hiera" in model_name)
                else None
            ),
        )
        self.model_name = model_name
        self.output_fmt = getattr(self.encoder, "output_fmt", "NHCW")
        num_features = self.encoder.feature_info.channels()
        self.neck_st1 = nn.Sequential(
            nn.Linear(num_features[-1] * 2, transformer_dim),
            nn.GELU(),
            nn.LayerNorm(transformer_dim),
        )
        self.neck_st2 = nn.Sequential(
            nn.Linear(num_features[-1] * 2, transformer_dim),
            nn.GELU(),
            nn.LayerNorm(transformer_dim),
        )
        self.neck_ax = nn.Sequential(
            nn.Linear(num_features[-1] * 2, transformer_dim),
            nn.GELU(),
            nn.LayerNorm(transformer_dim),
        )
        self.aux_head = nn.ModuleList(
            [
                nn.Sequential(
                    AttentionPooling(transformer_dim),
                    nn.Dropout(0.2),
                    nn.Linear(transformer_dim, num_class),
                )
                for _ in range(5)
            ]
        )

        self.pos_enc = SinusoidalPosEmb(dim=transformer_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim * 4,
                dropout=0.0,
                activation=nn.GELU(),
                batch_first=True,
                norm_first=False,
            ),
            transformer_num_layers,
        )
        self.head = nn.ModuleList(
            [
                nn.Sequential(
                    AttentionPooling(transformer_dim),
                    nn.Dropout(0.2),
                    nn.Linear(transformer_dim, num_class),
                )
                for _ in range(5)
            ]
        )
        self.transformer_dim = transformer_dim

    def forward_image_feats_st1(self, img_st1):
        img_st1 = img_st1[:, :, None]
        bs, seq_len, ch, h, w = img_st1.shape
        img_st1 = img_st1.reshape(bs * seq_len, ch, h, w)
        img_feats = self.encoder(img_st1)
        if self.output_fmt == "NHWC":
            img_feats = [
                img_feat.permute(0, 3, 1, 2).contiguous() for img_feat in img_feats
            ]
        img_feats = img_feats[-1]
        _, ch, h, w = img_feats.shape
        img_feats = img_feats.reshape(bs, seq_len, ch, h, w)  # (bs, seq_len, ch, h, w)
        img_feats = torch.cat(
            [img_feats.mean((3, 4)), img_feats.amax((3, 4))], 2
        )  # (bs, seq_len, ch * 2)
        img_feats = self.neck_st1(img_feats)
        return img_feats

    def forward_image_feats_st2(self, img_st2):
        img_st2 = img_st2[:, :, None]
        bs, seq_len, ch, h, w = img_st2.shape
        img_st2 = img_st2.reshape(bs * seq_len, ch, h, w)
        img_feats = self.encoder(img_st2)
        if self.output_fmt == "NHWC":
            img_feats = [
                img_feat.permute(0, 3, 1, 2).contiguous() for img_feat in img_feats
            ]
        img_feats = img_feats[-1]
        _, ch, h, w = img_feats.shape
        img_feats = img_feats.reshape(bs, seq_len, ch, h, w)  # (bs, seq_len, ch, h, w)
        img_feats = torch.cat(
            [img_feats.mean((3, 4)), img_feats.amax((3, 4))], 2
        )  # (bs, seq_len, ch * 2)
        img_feats = self.neck_st2(img_feats)
        return img_feats

    def forward_image_feats_ax(self, img_ax):
        img_ax = img_ax[:, :, None]
        bs, seq_len, ch, h, w = img_ax.shape
        img_ax = img_ax.reshape(bs * seq_len, ch, h, w)
        img_feats = self.encoder(img_ax)
        if self.output_fmt == "NHWC":
            img_feats = [
                img_feat.permute(0, 3, 1, 2).contiguous() for img_feat in img_feats
            ]
        img_feats = img_feats[-1]
        _, ch, h, w = img_feats.shape
        img_feats = img_feats.reshape(bs, seq_len, ch, h, w)  # (bs, seq_len, ch, h, w)
        img_feats = torch.cat(
            [img_feats.mean((3, 4)), img_feats.amax((3, 4))], 2
        )  # (bs, seq_len, ch * 2)
        img_feats = self.neck_ax(img_feats)
        return img_feats

    def forward_head(
        self, img_feats_st1, img_feats_st2, img_feats_ax, pad_mask, mask=None
    ):
        img_feats = torch.cat(
            [
                img_feats_st1,
                img_feats_st2,
                img_feats_ax,
            ],
            1,
        )  # (bs, seq_len * 2 + seq_len_ax, 256)
        # token mask augmentation
        if mask == None:
            if self.training:
                token_mask_rate = torch.rand(1) * self.max_token_mask_rate
                mask = torch.rand(img_feats.shape[:2]) < token_mask_rate
            else:
                mask = torch.zeros(img_feats.shape[:2]).bool()
        mask = mask.to(device=img_feats.device)
        mask = torch.logical_and(mask, pad_mask.bool())
        img_feats[mask] = 0
        pos_emb = torch.cat(
            [
                self.pos_enc(torch.arange(img_feats_st1.shape[1]))[None, :].to(
                    device=img_feats.device
                ),
                self.pos_enc(torch.arange(img_feats_st2.shape[1]))[None, :].to(
                    device=img_feats.device
                ),
                self.pos_enc(torch.arange(img_feats_ax.shape[1] // 2))[None, :].to(
                    device=img_feats.device
                ),
                self.pos_enc(torch.arange(img_feats_ax.shape[1] // 2))[None, :].to(
                    device=img_feats.device
                ),
            ],
            1,
        )
        img_feats = self.transformer(
            img_feats + pos_emb,
            src_key_padding_mask=mask,
        )
        h_input = dict(x=img_feats, mask=mask)
        logit = [h(h_input) for h in self.head]

        return logit

    def forward_aux_head(self, img_feats_st1, img_feats_st2, img_feats_ax, pad_mask):
        img_feats = torch.cat(
            [
                img_feats_st1,
                img_feats_st2,
                img_feats_ax,
            ],
            1,
        )
        h_input = dict(x=img_feats, mask=pad_mask.bool())
        aux_logit = [h(h_input) for h in self.aux_head]
        return aux_logit

    def forward(
        self,
        img_st1: torch.Tensor,
        img_st2: torch.Tensor,
        img_ax: torch.Tensor,
        pad_mask: torch.Tensor,
    ):
        """
        img: (bs, ch, h, w)
        """
        img_feats_st1 = self.forward_image_feats_st1(img_st1)
        img_feats_st2 = self.forward_image_feats_st2(img_st2)
        img_feats_ax = self.forward_image_feats_ax(img_ax)
        logit = self.forward_head(img_feats_st1, img_feats_st2, img_feats_ax, pad_mask)
        aux_logit = self.forward_aux_head(
            img_feats_st1, img_feats_st2, img_feats_ax, pad_mask
        )
        return {"logit": logit, "aux_logit": aux_logit}


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


class RSNA2024Stage2LightningModel(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool = False,
        num_class: int = 3,
        model_name: str = "resnet34",
        drop_path_rate: float = 0,
        img_size_st1: int = 128,
        img_size_st2: int = 128,
        img_size_ax: int = 128,
        transformer_dim: int = 256,
        transformer_num_layers: int = 2,
        transformer_nhead: int = 8,
        max_token_mask_rate: float = 0.0,
        mixup_p: float = 0.0,
        mixup_alpha: float = 0.5,
        no_mixup_epochs: int = 0,
        lr: float = 1e-3,
        backbone_lr: float = None,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.__build_model(
            pretrained=pretrained,
            num_class=num_class,
            model_name=model_name,
            drop_path_rate=drop_path_rate,
            img_size_st1=img_size_st1,
            img_size_st2=img_size_st2,
            img_size_ax=img_size_ax,
            transformer_dim=transformer_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_nhead=transformer_nhead,
            max_token_mask_rate=max_token_mask_rate,
        )
        self.mixupper = Mixup(p=mixup_p, alpha=mixup_alpha)
        self.hparams.backbone_lr = (
            self.hparams.backbone_lr if self.hparams.backbone_lr is not None else lr
        )
        self.gt_val = []
        self.logit_val = []

    def __build_model(
        self,
        pretrained: bool = False,
        num_class: int = 3,
        model_name: str = "resnet34",
        drop_path_rate: float = 0,
        img_size_st1: int = 128,
        img_size_st2: int = 128,
        img_size_ax: int = 128,
        transformer_dim: int = 256,
        transformer_num_layers: int = 2,
        transformer_nhead: int = 8,
        max_token_mask_rate: float = 0.0,
    ):
        self.model = RSNA2024Stage2Model(
            pretrained=pretrained,
            num_class=num_class,
            model_name=model_name,
            drop_path_rate=drop_path_rate,
            img_size_st1=img_size_st1,
            img_size_st2=img_size_st2,
            img_size_ax=img_size_ax,
            transformer_dim=transformer_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_nhead=transformer_nhead,
            max_token_mask_rate=max_token_mask_rate,
        )
        self.model_ema = ModelEmaV3(
            self.model,
            decay=0.999,
            update_after_step=100,
            use_warmup=True,
            warmup_power=3 / 4,
        )
        self.criterions = {
            "ce": nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 2.0, 4.0], dtype=torch.float)
            ),
        }

    def calc_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        losses = {}
        # smooth = 0.1
        # true = labels["targets"] * (1 - (smooth / 0.5)) + smooth
        self.criterions["ce"].weight = self.criterions["ce"].weight.to(
            outputs["logit"][0].device
        )

        losses["ce_l"] = self.criterions["ce"](
            outputs["logit"][0], labels["label"][:, 0]
        )
        losses["ce_r"] = self.criterions["ce"](
            outputs["logit"][1], labels["label"][:, 1]
        )
        losses["ce"] = self.criterions["ce"](outputs["logit"][2], labels["label"][:, 2])
        losses["ce_l_ax"] = self.criterions["ce"](
            outputs["logit"][3], labels["label"][:, 3]
        )
        losses["ce_r_ax"] = self.criterions["ce"](
            outputs["logit"][4], labels["label"][:, 4]
        )

        losses["ce_l_aux"] = self.criterions["ce"](
            outputs["aux_logit"][0], labels["label"][:, 0]
        )
        losses["ce_r_aux"] = self.criterions["ce"](
            outputs["aux_logit"][1], labels["label"][:, 1]
        )
        losses["ce_aux"] = self.criterions["ce"](
            outputs["aux_logit"][2], labels["label"][:, 2]
        )
        losses["ce_l_ax_aux"] = self.criterions["ce"](
            outputs["aux_logit"][3], labels["label"][:, 3]
        )
        losses["ce_r_ax_aux"] = self.criterions["ce"](
            outputs["aux_logit"][4], labels["label"][:, 4]
        )
        losses["main"] = (
            losses["ce_l"]
            + losses["ce_r"]
            + losses["ce"]
            + losses["ce_l_ax"]
            + losses["ce_r_ax"]
        ) / 5
        losses["aux"] = (
            losses["ce_l_aux"]
            + losses["ce_r_aux"]
            + losses["ce_aux"]
            + losses["ce_l_ax_aux"]
            + losses["ce_r_ax_aux"]
        ) / 5
        losses["loss"] = (losses["main"] + losses["aux"]) / 2
        return losses

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model)
        step_output = {}
        outputs = {}
        loss_target = {}
        self.mixupper.init_lambda()
        (
            volume_st1,
            volume_st2,
            volume_ax,
            pad_mask_st1,
            pad_mask_st2,
            pad_mask_ax,
            label,
        ) = (
            batch["volume_st1"],
            batch["volume_st2"],
            batch["volume_ax"],
            batch["pad_mask_st1"],
            batch["pad_mask_st2"],
            batch["pad_mask_ax"],
            batch["label"],
        )
        pad_mask = torch.cat([pad_mask_st1, pad_mask_st2, pad_mask_ax], 1)
        if (
            self.mixupper.do_mixup
            and self.current_epoch
            < self.trainer.max_epochs - self.hparams.no_mixup_epochs
        ):
            volume_st1 = self.mixupper.lam * volume_st1 + (
                1 - self.mixupper.lam
            ) * volume_st1.flip(0)
            volume_st2 = self.mixupper.lam * volume_st2 + (
                1 - self.mixupper.lam
            ) * volume_st2.flip(0)
            volume_ax = self.mixupper.lam * volume_ax + (
                1 - self.mixupper.lam
            ) * volume_ax.flip(0)
            pad_mask += pad_mask.flip(0).clamp(0, 1)
        outputs.update(self.model(volume_st1, volume_st2, volume_ax, pad_mask))
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
        loss_dict = {}
        for key in losses:
            if key == "loss":
                loss_dict["train_loss"] = losses[key]
            else:
                loss_dict[f"train_{key}_loss"] = losses[key]
        self.log_dict(
            loss_dict,
            sync_dist=True,
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = {}
        outputs = {}
        loss_target = {}

        (
            volume_st1,
            volume_st2,
            volume_ax,
            pad_mask_st1,
            pad_mask_st2,
            pad_mask_ax,
            label,
        ) = (
            batch["volume_st1"],
            batch["volume_st2"],
            batch["volume_ax"],
            batch["pad_mask_st1"],
            batch["pad_mask_st2"],
            batch["pad_mask_ax"],
            batch["label"],
        )
        pad_mask = torch.cat([pad_mask_st1, pad_mask_st2, pad_mask_ax], 1)
        outputs.update(
            self.model_ema.module(volume_st1, volume_st2, volume_ax, pad_mask)
        )
        loss_target["label"] = label
        losses = self.calc_loss(outputs, loss_target)

        logit = (
            torch.softmax(torch.stack(outputs["logit"], 1), -1).detach().cpu().numpy()
        )  # (bs, 5, 3)
        self.logit_val.append(logit)
        self.gt_val.append(label.detach().cpu().numpy())

        step_output.update(losses)

        loss_dict = {}
        for key in losses:
            if key == "loss":
                loss_dict["val_loss"] = losses[key]
            else:
                loss_dict[f"val_{key}_loss"] = losses[key]
        self.log_dict(
            loss_dict,
            sync_dist=True,
        )
        return step_output

    def on_validation_epoch_end(self):
        conditions = [
            "nfn_l",
            "nfn_r",
            "scs",
            "ss_l",
            "ss_r",
        ]
        weights = [1, 2, 4]
        logit_val = np.concatenate(self.logit_val)  # (bs, 5, 3)
        gt_val = np.concatenate(self.gt_val)  # (bs, 5)
        bs, conds = gt_val.shape
        gt_val_onehot = np.zeros((bs * conds, 3))
        for i, gt in enumerate(gt_val.reshape(-1)):
            gt_val_onehot[i, gt] = 1
        gt_val_onehot = gt_val_onehot.reshape(bs, conds, 3)
        logloss = {}
        for i, c in enumerate(conditions):
            condition_loss = sklearn.metrics.log_loss(
                y_true=gt_val_onehot[:, i],
                y_pred=logit_val[:, i],
                sample_weight=np.asarray(
                    [weights[j] for j in gt_val_onehot[:, i].argmax(-1)]
                ),
            )
            logloss["val_" + c] = condition_loss
        logloss["val_logloss"] = (
            (logloss["val_nfn_l"] + logloss["val_nfn_r"]) / 2
            + logloss["val_scs"]
            + (logloss["val_ss_l"] + logloss["val_ss_r"]) / 2
        ) / 3
        self.logit_val.clear()
        self.gt_val.clear()

        self.log_dict(
            logloss,
            sync_dist=True,
        )

    def get_optimizer_parameters(self):
        no_decay = ["bias", "gamma", "beta"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in list(self.model.encoder.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.hparams.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.encoder.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.neck_st1.named_parameters())
                    + list(self.model.neck_st2.named_parameters())
                    + list(self.model.neck_ax.named_parameters())
                    + list(self.model.aux_head.named_parameters())
                    + list(self.model.transformer.named_parameters())
                    + list(self.model.head.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.hparams.lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.neck_st1.named_parameters())
                    + list(self.model.neck_st2.named_parameters())
                    + list(self.model.neck_ax.named_parameters())
                    + list(self.model.aux_head.named_parameters())
                    + list(self.model.transformer.named_parameters())
                    + list(self.model.head.named_parameters())
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
        parser = parent_parser.add_argument_group("RSNA2024Stage2LightningModel")
        parser.add_argument(
            "--num_class",
            default=3,
            type=int,
            metavar="OCL",
            dest="num_class",
        )
        parser.add_argument(
            "--model_name",
            default="resnet34",
            type=str,
            metavar="MNS",
            help="Name (as in ``timm``) of the feature extractor",
            dest="model_name",
        )
        parser.add_argument(
            "--drop_path_rate",
            default=None,
            type=float,
            metavar="DPRS",
            dest="drop_path_rate",
        )
        parser.add_argument(
            "--transformer_dim",
            default=256,
            type=int,
            metavar="TD",
            dest="transformer_dim",
        )
        parser.add_argument(
            "--transformer_num_layers",
            default=2,
            type=int,
            metavar="TNL",
            dest="transformer_num_layers",
        )
        parser.add_argument(
            "--transformer_nhead",
            default=8,
            type=int,
            metavar="TNH",
            dest="transformer_nhead",
        )
        parser.add_argument(
            "--max_token_mask_rate",
            default=0.0,
            type=float,
            metavar="MTMR",
            dest="max_token_mask_rate",
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
            metavar="BLRS",
            help="initial learning rate",
            dest="backbone_lr",
        )
        parser.add_argument(
            "--weight_decay",
            default=1e-2,
            type=float,
            metavar="WD",
            help="initial weight decay",
            dest="weight_decay",
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
        default="32-true",
    )
    parent_parser.add_argument(
        "--img_size_st1",
        default=128,
        type=int,
        metavar="ISZST1",
        dest="img_size_st1",
    )
    parent_parser.add_argument(
        "--in_chans_st1",
        default=20,
        type=int,
        metavar="ICHST1",
        dest="in_chans_st1",
    )
    parent_parser.add_argument(
        "--img_size_st2",
        default=128,
        type=int,
        metavar="ISZST2",
        dest="img_size_st2",
    )
    parent_parser.add_argument(
        "--in_chans_st2",
        default=20,
        type=int,
        metavar="ICHST2",
        dest="in_chans_st2",
    )
    parent_parser.add_argument(
        "--in_chans_ax",
        default=5,
        type=int,
        metavar="ICHA",
        dest="in_chans_ax",
    )
    parent_parser.add_argument(
        "--img_size_ax",
        default=128,
        type=int,
        metavar="ISZA",
        dest="img_size_ax",
    )
    parser = RSNA2024Stage2LightningModel.add_model_specific_args(parent_parser)
    parser = RSNA2024Stage2DataModule.add_model_specific_args(parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed, workers=True)
    if not args.debug:
        warnings.simplefilter("ignore")
    assert args.fold < 5
    train_coord_df = pd.read_csv("train_coord_df.csv")
    for fold in range(5):
        if fold != args.fold:
            continue
        train_df = train_coord_df[train_coord_df.fold_id != fold].reset_index(drop=True)
        valid_df = train_coord_df[train_coord_df.fold_id == fold].reset_index(drop=True)
        datamodule = RSNA2024Stage2DataModule(
            train_df=train_df,
            valid_df=valid_df,
            img_size_st1=args.img_size_st1,
            in_chans_st1=args.in_chans_st1,
            crop_range_st1=args.crop_range_st1,
            img_size_st2=args.img_size_st2,
            in_chans_st2=args.in_chans_st2,
            crop_range_st2=args.crop_range_st2,
            img_size_ax=args.img_size_ax,
            in_chans_ax=args.in_chans_ax,
            crop_range_ax=args.crop_range_ax,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        model = RSNA2024Stage2LightningModel(
            pretrained=True,
            num_class=args.num_class,
            model_name=args.model_name,
            drop_path_rate=args.drop_path_rate,
            img_size_st1=args.img_size_st1,
            img_size_st2=args.img_size_st2,
            img_size_ax=args.img_size_ax,
            transformer_dim=args.transformer_dim,
            transformer_num_layers=args.transformer_num_layers,
            transformer_nhead=args.transformer_nhead,
            max_token_mask_rate=args.max_token_mask_rate,
            mixup_p=args.mixup_p,
            mixup_alpha=args.mixup_alpha,
            no_mixup_epochs=args.no_mixup_epochs,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
        )

        logdir = f"../../../logs/stage2/exp{EXP_ID}/{args.logdir}/Sagittal_t1+Sagittal_t2-STIR+Axial_T2/fold{fold}"
        print(f"logdir = {logdir}")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_logloss",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
            mode="min",
        )
        os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
        if not args.debug:
            wandb_logger = WandbLogger(
                name=f"stage2/exp{EXP_ID}/{args.logdir}/Sagittal_t1+Sagittal_t2-STIR+Axial_T2/fold{fold}",
                tags=[f"fold{fold}", "stage2", "Sagittal_t1+Sagittal_t2-STIR+Axial_T2"],
                save_dir=logdir,
                project="rsna-2024-lumbar-spine-degenerative-classification",
                notes=COMMENT,
            )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_logloss", patience=10, log_rank_zero_only=True
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
