import sys

sys.path.append("../../")
sys.path.append("../../stage1/exp017")
import argparse
import os
from glob import glob

import albumentations as A
import numpy as np
import pandas as pd
import pydicom
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold
from stage1.exp017.train import RSNA2024Stage1LightningModel
from torch.nn import functional as F
from tqdm.auto import tqdm
from train import EXP_ID, RSNA2024Stage2LightningModel, crop_axial, crop_sagittal


def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype
        pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift

    intercept = float(dcm.RescaleIntercept) if hasattr(dcm, "RescaleIntercept") else 0
    slope = float(dcm.RescaleSlope) if hasattr(dcm, "RescaleSlope") else 1
    center = int(dcm.WindowCenter)
    width = int(dcm.WindowWidth)
    low = center - width / 2
    high = center + width / 2

    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.clip(pixel_array, low, high)
    return pixel_array


def get_ckpt_query(stage, exp_id, series_descriptions, fold, logdir, log_root):
    query = ""
    query += f"stage{stage}/"
    query += f"exp{exp_id}/"
    if logdir != None:
        query += f"{logdir}/"
    else:
        query += "**/"
    if series_descriptions != None:
        query += f"{series_descriptions}/"
    else:
        query += "**/"
    if fold != None:
        query += f"fold{fold}/"
    else:
        query += "**/"
    query += "**/best_loss.ckpt"
    ckpt_paths = sorted(
        list(
            set(
                glob(
                    os.path.join(
                        log_root,
                        query,
                    ),
                    recursive=True,
                )
            )
        )
    )
    return ckpt_paths


def get_models(
    stage: str = "1",
    exp_id: str = "017",
    series_description: str = "Sagittal_T1",
    fold: int = None,
    logdirs=None,
    log_root: str = "../../../logs",
    device="cpu",
):
    if logdirs == None:
        ckpt_paths = get_ckpt_query(
            stage=stage,
            exp_id=exp_id,
            series_descriptions=series_description,
            fold=fold,
            logdir=None,
            log_root=log_root,
        )
    else:
        ckpt_paths = []
        for logdir in logdirs:
            ckpt_paths += get_ckpt_query(
                stage=stage,
                exp_id=exp_id,
                series_descriptions=series_description,
                fold=fold,
                logdir=logdir,
                log_root=log_root,
            )
    for ckpt_path in ckpt_paths:
        print("loading ckpt: ", ckpt_path)
    if stage == "1":
        models = [
            RSNA2024Stage1LightningModel.load_from_checkpoint(
                ckpt_path, pretrained=False, map_location=torch.device("cpu")
            )
            .model_ema.module.to(device)
            .eval()
            for ckpt_path in ckpt_paths
        ]
    else:
        models = [
            RSNA2024Stage2LightningModel.load_from_checkpoint(
                ckpt_path, pretrained=False, map_location=torch.device("cpu")
            )
            .model_ema.module.to(device)
            .eval()
            for ckpt_path in ckpt_paths
        ]
    return models, ckpt_paths


def get_volume(
    series_id: str,
    description: str,
    dicoms_root: str = "../../../input/rsna-2024-lumbar-spine-degenerative-classification/train_images",
):
    volume = []
    img_paths = glob(os.path.join(dicoms_root, f"**/{series_id}/*.dcm"))
    if description == "Axial T2":
        planes = 2
    else:
        planes = 0
    positions = np.asarray(
        [
            float(pydicom.dcmread(img_path).ImagePositionPatient[planes])
            for img_path in img_paths
        ]
    )
    idx = np.argsort(-positions)
    img_paths = np.asarray(img_paths)[idx]
    for img_path in img_paths:
        dicom = pydicom.dcmread(img_path)
        img = standardize_pixel_array(dicom)
        volume.append(img)
    if len(set([img.shape[0] for img in volume])) > 1:
        for i in range(len(volume)):
            volume[i] = F.interpolate(
                torch.Tensor(volume[i])[None, None, ...],
                size=volume[0].shape,
                align_corners=True,
                mode="bilinear",
            ).numpy()[0, 0]
    volume = np.stack(volume)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
    # volume = (volume * 255).astype(np.uint8)
    return volume


def preprocess_stage1_sagittal(volume):
    volume = F.interpolate(
        torch.Tensor(volume)[None, None, ...],
        size=(20, 256, 256),
        mode="trilinear",
        align_corners=True,
    ).numpy()[0, 0]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
    volume = (volume * 255).astype(np.uint8)
    transform = A.Compose(
        [ToTensorV2()],
    )
    transformed = transform(
        image=volume.transpose((1, 2, 0)),
    )
    volume = transformed["image"] / 255.0  # (c, h, w)
    return volume[None, :]  # (1, c, h, w)


def preprocess_stage2_axial(
    volume, keypoints, img_size: int = 256, in_chans: int = 3, crop_range: float = 2
):
    # keypoints: (levels, [x_l, y_l, x_r, x_l, instance_number])
    crop_volumes = []
    transform = A.Compose(
        [A.Resize(img_size, img_size), ToTensorV2()],
    )
    mask = torch.zeros((len(keypoints), in_chans * 2))
    for i in range(len(keypoints)):
        if keypoints[i][0] == -1:
            cropped_volume = torch.zeros(
                (in_chans * 2, img_size, img_size), dtype=torch.float32
            )
            crop_volumes.append(cropped_volume)
            mask[i] = 1
            continue
        try:
            d = (
                np.sqrt(
                    (keypoints[i][0] - keypoints[i][2]) ** 2
                    + (keypoints[i][1] - keypoints[i][3]) ** 2
                )
                * crop_range
            )
            cropped_volume_l, pad_mask_l = crop_axial(
                volume,
                img_size,
                in_chans,
                keypoints[i][:2],
                keypoints[i][4],
                d,
                transform,
            )
            mask[i, :in_chans] = pad_mask_l
        except:
            cropped_volume_l = torch.zeros(
                (in_chans, img_size, img_size), dtype=torch.float32
            )
            mask[i, :in_chans] = 1

        try:
            d = (
                np.sqrt(
                    (keypoints[i][0] - keypoints[i][2]) ** 2
                    + (keypoints[i][1] - keypoints[i][3]) ** 2
                )
                * crop_range
            )
            cropped_volume_r, pad_mask_r = crop_axial(
                volume,
                img_size,
                in_chans,
                keypoints[i][2:],
                keypoints[i][4],
                d,
                transform,
            )
            mask[i, in_chans:] = pad_mask_r
        except:
            cropped_volume_r = torch.zeros(
                (in_chans, img_size, img_size), dtype=torch.float32
            )
            mask[i, in_chans:] = 1
        cropped_volume = torch.cat([cropped_volume_l, cropped_volume_r], 0)
        crop_volumes.append(cropped_volume)

    crop_volumes = torch.stack(crop_volumes)
    # print(crop_volumes.shape)
    return crop_volumes, mask  # (5, c, h, w)


def preprocess_stage2_sagittal(
    volume, keypoints, img_size: int = 128, in_chans: int = 20, crop_range: float = 1.2
):
    # keypoints: (levels, 2)
    crop_volumes = []
    transform = A.Compose(
        [A.Resize(img_size, img_size), ToTensorV2()],
    )
    mask = torch.zeros((len(keypoints), in_chans))
    for i in range(len(keypoints)):
        if keypoints[i][0] == -1:
            cropped_volume = torch.zeros(
                (in_chans, img_size, img_size), dtype=torch.float32
            )
            crop_volumes.append(cropped_volume)
            mask[i] = 1
            continue
        try:
            dists = []
            if i != 0 and (keypoints[i - 1][0] != -1):
                dists.append(np.sqrt(np.sum((keypoints[i] - keypoints[i - 1]) ** 2)))
            if i != len(keypoints) - 1 and (keypoints[i + 1][0] != -1):
                dists.append(np.sqrt(np.sum((keypoints[i] - keypoints[i + 1]) ** 2)))
            d = sum(dists) / len(dists) * crop_range
            cropped_volume, pad_mask = crop_sagittal(
                volume, img_size, in_chans, keypoints[i], d, transform
            )
            mask[i] = pad_mask
        except:
            cropped_volume = torch.zeros(
                (in_chans, img_size, img_size), dtype=torch.float32
            )
            mask[i] = 1
        crop_volumes.append(cropped_volume)
    crop_volumes = torch.stack(crop_volumes)
    return crop_volumes, mask  # (5, c, h, w)


def prediction_stage1_sagittal(models, volume, threshold=0.3):
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = [model(volume) for model in models]
        logit = np.stack(
            [output["logit"].detach().cpu().sigmoid()[0].numpy() for output in outputs]
        ).mean(0)

    heatmap = logit
    heatmap = heatmap * (heatmap > threshold)
    keypoints = -np.ones((5, 2))
    for c, h in enumerate(heatmap):
        if h.max() < threshold:
            continue
        y = h.sum(1).argmax() / h.shape[0]
        x = h.sum(0).argmax() / h.shape[1]
        keypoints[c] = np.asarray([x, y])
    return keypoints


def feature_extract_st1_stage2(models, volume, mask):
    # volume: (level, c, h, w)
    # mask: (level, c)
    masked_features = [
        torch.zeros((volume.shape[0], volume.shape[1], model.transformer_dim)).to(
            device=volume.device, dtype=volume.dtype
        )
        for model in models
    ]
    level_mask = mask.sum(1) == mask.shape[1]  # (level)
    if level_mask.sum() < len(level_mask):
        volume = volume[~level_mask.bool()]
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = [
                    model.forward_image_feats_st1(volume).detach() for model in models
                ]
        for i, f in enumerate(features):
            masked_features[i][~level_mask.bool()] = f
    return masked_features  # (level, c, d)


def feature_extract_st2_stage2(models, volume, mask):
    masked_features = [
        torch.zeros((volume.shape[0], volume.shape[1], model.transformer_dim)).to(
            device=volume.device, dtype=volume.dtype
        )
        for model in models
    ]
    level_mask = mask.sum(1) == mask.shape[1]  # (level)
    if level_mask.sum() < len(level_mask):
        volume = volume[~level_mask.bool()]
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = [
                    model.forward_image_feats_st2(volume).detach() for model in models
                ]
        for i, f in enumerate(features):
            masked_features[i][~level_mask.bool()] = f
    return masked_features


def feature_extract_ax_stage2(models, volume, mask):
    masked_features = [
        torch.zeros((volume.shape[0], volume.shape[1], model.transformer_dim)).to(
            device=volume.device, dtype=volume.dtype
        )
        for model in models
    ]
    level_mask = mask.sum(1) == mask.shape[1]  # (level)
    if level_mask.sum() < len(level_mask):
        volume = volume[~level_mask.bool()]
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = [
                    model.forward_image_feats_ax(volume).detach() for model in models
                ]
        for i, f in enumerate(features):
            masked_features[i][~level_mask.bool()] = f
    return masked_features


def forward_head_stage2(model, features_st1, features_st2, features_ax, mask):
    # features_st1, features_st2, features_ax: (level, c, d)
    # mask: (level, c)
    level_mask = mask.sum(1) == mask.shape[1]  # (level)
    masked_preds = torch.ones((5, 5, 3)).numpy() / 3  # (conds, level, 3)
    if level_mask.sum() < len(level_mask):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model.forward_head(
                    features_st1[~level_mask],
                    features_st2[~level_mask],
                    features_ax[~level_mask],
                    mask[~level_mask],
                )
            preds = np.stack(
                [p.softmax(1).detach().cpu().numpy() for p in outputs]
            )  # (5, level, 3)
        masked_preds[:, ~level_mask.cpu().numpy()] = preds
    return masked_preds


def inference_pipeline_one_study(
    series_df: pd.DataFrame,
    model_stage1_sagittal_t1,
    model_stage1_sagittal_t2,
    models_stage2,
    axial_keypoints_df,
    dicoms_root: str = "../../../input/rsna-2024-lumbar-spine-degenerative-classification/train_images",
    in_chans_st1: int = 20,
    img_size_st1: int = 128,
    crop_range_st1: float = 1.0,
    in_chans_st2: int = 20,
    img_size_st2: int = 128,
    crop_range_st2: float = 1.0,
    in_chans_ax: int = 3,
    img_size_ax: int = 128,
    crop_range_ax: float = 1.0,
    device: str = "cuda:0",
):

    series_ids = series_df[
        series_df.series_description == "Sagittal T1"
    ].series_id.to_list()
    st1_features = []
    st1_masks = []
    for series_id in series_ids:
        volume = get_volume(
            series_id=series_id, description="Sagittal T1", dicoms_root=dicoms_root
        )
        stage1_volume = preprocess_stage1_sagittal(volume).to(device)
        keypoints = prediction_stage1_sagittal(model_stage1_sagittal_t1, stage1_volume)
        cropped_volume, mask = preprocess_stage2_sagittal(
            volume,
            keypoints,
            img_size=img_size_st1,
            in_chans=in_chans_st1,
            crop_range=crop_range_st1,
        )
        cropped_volume = cropped_volume.to(device)
        mask = mask.to(device)
        feature = feature_extract_st1_stage2(models_stage2, cropped_volume, mask)
        st1_features.append(feature)
        st1_masks.append(mask)
    if len(series_ids) == 0:
        st1_features.append(
            [
                torch.zeros(
                    (5, in_chans_st1, model.transformer_dim),
                    device=device,
                    dtype=torch.float32,
                )
                for model in models_stage2
            ]
        )
        st1_masks.append(torch.zeros((5, in_chans_st1), device=device))

    series_ids = series_df[
        series_df.series_description == "Sagittal T2/STIR"
    ].series_id.to_list()
    st2_features = []
    st2_masks = []
    for series_id in series_ids:
        volume = get_volume(
            series_id=series_id, description="Sagittal T2/STIR", dicoms_root=dicoms_root
        )
        stage1_volume = preprocess_stage1_sagittal(volume).to(device)
        keypoints = prediction_stage1_sagittal(model_stage1_sagittal_t2, stage1_volume)
        cropped_volume, mask = preprocess_stage2_sagittal(
            volume,
            keypoints,
            img_size=img_size_st2,
            in_chans=in_chans_st2,
            crop_range=crop_range_st2,
        )
        cropped_volume = cropped_volume.to(device)
        mask = mask.to(device)
        feature = feature_extract_st2_stage2(models_stage2, cropped_volume, mask)
        st2_features.append(feature)
        st2_masks.append(mask)
    if len(series_ids) == 0:
        st2_features.append(
            [
                torch.zeros(
                    (5, in_chans_st2, model.transformer_dim),
                    device=device,
                    dtype=torch.float32,
                )
                for model in models_stage2
            ]
        )
        st2_masks.append(torch.zeros((5, in_chans_st2), device=device))

    series_ids = series_df[
        series_df.series_description == "Axial T2"
    ].series_id.to_list()
    ax_features = []
    ax_masks = []
    for series_id in series_ids:
        volume = get_volume(
            series_id=series_id, description="Axial T2", dicoms_root=dicoms_root
        )
        level_map = {"L1/L2": 0, "L2/L3": 1, "L3/L4": 2, "L4/L5": 3, "L5/S1": 4}
        keypoint_axial = (
            axial_keypoints_df[axial_keypoints_df.series_id == series_id]
            .sort_values("level")
            .loc[:, ["level", "x_l", "x_r", "y_l", "y_r", "instance_number"]]
            .values
        )
        keypoints = -np.zeros((5, 5))
        for k in keypoint_axial:
            keypoints[level_map[k[0]]] = k[1:6]
        cropped_volume, mask = preprocess_stage2_axial(
            volume,
            keypoints,
            img_size=img_size_ax,
            in_chans=in_chans_ax,
            crop_range=crop_range_ax,
        )
        cropped_volume = cropped_volume.to(device)
        mask = mask.to(device)
        feature = feature_extract_ax_stage2(models_stage2, cropped_volume, mask)
        ax_features.append(feature)
        ax_masks.append(mask)
    if len(series_ids) == 0:
        ax_features.append(
            [
                torch.zeros(
                    (5, in_chans_ax * 2, model.transformer_dim),
                    device=device,
                    dtype=torch.float32,
                )
                for model in models_stage2
            ]
        )
        ax_masks.append(torch.zeros((5, in_chans_ax * 2), device=device))
    # st1_features: (num_feats, models, ...)
    preds = []
    for st1_feat, st1_mask in zip(st1_features, st1_masks):
        for st2_feat, st2_mask in zip(st2_features, st2_masks):
            for ax_feat, ax_mask in zip(ax_features, ax_masks):
                for model, st1_f, st2_f, ax_f in zip(
                    models_stage2, st1_feat, st2_feat, ax_feat
                ):
                    preds.append(
                        forward_head_stage2(
                            model,
                            st1_f,
                            st2_f,
                            ax_f,
                            torch.cat([st1_mask, st2_mask, ax_mask], 1).bool(),
                        )
                    )
    preds = np.stack(preds).mean(0)
    return preds


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=True)
    parent_parser.add_argument("--fold", nargs="+", default=[0, 1, 2, 3, 4], type=int)
    parent_parser.add_argument(
        "--stage2_logdirs",
        nargs="+",
        default=[
            "caformer_s18_20x1x128x128_x1.0_20x1x128x128_x1.0_3x1x128x128_x1.0_mixup",
            "resnetrs50_20x1x128x128_x1.0_20x1x128x128_x1.0_3x1x128x128_x1.0_mixup",
        ],
    )
    parent_parser.add_argument(
        "--stage1_exp_id",
        default="017",
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
        "--crop_range_st1",
        default=1,
        type=float,
        metavar="CRST1",
        dest="crop_range_st1",
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
        "--crop_range_st2",
        default=1,
        type=float,
        metavar="CRST2",
        dest="crop_range_st2",
    )
    parent_parser.add_argument(
        "--in_chans_ax",
        default=3,
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
    parent_parser.add_argument(
        "--crop_range_ax",
        default=1,
        type=float,
        metavar="CRSA",
        dest="crop_range_ax",
    )
    parent_parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parent_parser.add_argument(
        "--out",
        default="eval_results.csv",
    )
    return parent_parser.parse_args()


def main(args):
    ###### dummy ######
    axial_keypoints_df = pd.concat(
        [
            pd.read_csv("../../../input/axial_val_keypoint_preds_fold0.csv"),
            pd.read_csv("../../../input/axial_val_keypoint_preds_fold1.csv"),
            pd.read_csv("../../../input/axial_val_keypoint_preds_fold2.csv"),
            pd.read_csv("../../../input/axial_val_keypoint_preds_fold3.csv"),
            pd.read_csv("../../../input/axial_val_keypoint_preds_fold4.csv"),
        ]
    ).reset_index(drop=True)
    axial_keypoints_df = axial_keypoints_df.rename(
        columns={
            "right_x": "x_l",
            "right_y": "y_l",
            "left_x": "x_r",
            "left_y": "y_r",
            "part_id": "level",
        }
    ).replace(
        {"level": {0: "L1/L2", 1: "L2/L3", 2: "L3/L4", 3: "L4/L5", 4: "L5/S1"}},
        inplace=False,
    )
    axial_keypoints_df.loc[:, ["x_l", "x_r", "y_l", "y_r"]] = (
        axial_keypoints_df.loc[:, ["x_l", "x_r", "y_l", "y_r"]] / 512
    )
    ###### dummy ######

    series_description_df = pd.read_csv(
        "../../../input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv"
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
    series_description_df = series_description_df.merge(
        train_df.loc[:, ["study_id", "fold_id"]], on=["study_id"], how="left"
    )
    preds_study_id = {}
    for fold in range(5):
        if fold not in args.fold:
            continue
        model_stage1_sagittal_t1, _ = get_models(
            stage="1",
            exp_id=args.stage1_exp_id,
            series_description="Sagittal_T1",
            fold=fold,
            logdirs=None,
            device=args.device,
        )
        model_stage1_sagittal_t2, _ = get_models(
            stage="1",
            exp_id=args.stage1_exp_id,
            series_description="Sagittal_T2-STIR",
            fold=fold,
            logdirs=None,
            device=args.device,
        )
        models_stage2, _ = get_models(
            stage="2",
            exp_id=EXP_ID,
            series_description=None,
            fold=fold,
            logdirs=args.stage2_logdirs,
            device=args.device,
        )

        for study_id in tqdm(train_df[train_df.fold_id == fold].study_id.to_numpy()):
            series_df = series_description_df[
                series_description_df.study_id == study_id
            ]
            preds = inference_pipeline_one_study(
                series_df,
                model_stage1_sagittal_t1,
                model_stage1_sagittal_t2,
                models_stage2,
                axial_keypoints_df,
                dicoms_root="../../../input/rsna-2024-lumbar-spine-degenerative-classification/train_images",
                in_chans_st1=args.in_chans_st1,
                img_size_st1=args.img_size_st1,
                crop_range_st1=args.crop_range_st1,
                in_chans_st2=args.in_chans_st2,
                img_size_st2=args.img_size_st2,
                crop_range_st2=args.crop_range_st2,
                in_chans_ax=args.in_chans_ax,
                img_size_ax=args.img_size_ax,
                crop_range_ax=args.crop_range_ax,
                device=args.device,
            )
            preds_study_id[study_id] = preds
    row_id = []
    pred_values = []
    conditions = [
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "spinal_canal_stenosis",
        "left_subarticular_stenosis",
        "right_subarticular_stenosis",
    ]
    levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
    for study_id in preds_study_id.keys():
        for c in conditions:
            for l in levels:
                row_id.append(f"{study_id}_{c}_{l}")
        pred_values.append(preds_study_id[study_id].reshape(25, 3))
    pred_values = np.concatenate(pred_values)  # (n, 3)

    preds_df = (
        pd.DataFrame(
            pred_values, index=row_id, columns=["normal_mild", "moderate", "severe"]
        )
        .reset_index()
        .rename(columns={"index": "row_id"})
    )
    preds_df = (
        preds_df.fillna(1 / 3, inplace=False)
        .sort_values("row_id", ascending=True)
        .reset_index(drop=True)
    )
    print(preds_df.head())
    preds_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main(get_args())
