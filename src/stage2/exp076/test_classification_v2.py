import argparse
import os
import sys
from collections import defaultdict
from glob import glob

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
from albumentations.pytorch import ToTensorV2
from evaluation import (
    EXP_ID,
    feature_extract_ax_stage2,
    feature_extract_st1_stage2,
    feature_extract_st2_stage2,
    forward_head_stage2,
    get_models,
    get_volume,
    preprocess_stage2_axial,
    preprocess_stage2_sagittal,
)
from torch.nn import functional as F
from tqdm.auto import tqdm


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=True)
    parent_parser.add_argument("--log_root", default="../../../logs")
    parent_parser.add_argument("--fold", nargs="+", default=[0, 1, 2, 3, 4], type=int)
    parent_parser.add_argument(
        "--stage2_logdirs",
        nargs="+",
        default=[
            "caformer_s18_20x1x128x128_x1.0_20x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1",
            "resnetrs50_20x1x128x128_x1.0_20x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1",
        ],
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
    parent_parser.add_argument(
        "--crop_range_ax",
        default=1,
        type=float,
        metavar="CRSA",
        dest="crop_range_ax",
    )
    parent_parser.add_argument(
        "--dicoms_root",
        default="../../../input/rsna-2024-lumbar-spine-degenerative-classification/train_images",
    )
    parent_parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parent_parser.add_argument(
        "--series_description_df",
        default="../../../input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv",
    )
    parent_parser.add_argument(
        "--sagittal_keypoints_df",
        default="sagittal_keypoints_df.csv",
    )
    parent_parser.add_argument(
        "--axial_keypint_df",
        default="axial_keypoints_df.csv",
    )
    return parent_parser.parse_args()


def main(args):
    series_description_df = pd.read_csv(args.series_description_df)
    device = args.device
    sagittal_keypoints_df = pd.read_csv(args.sagittal_keypoints_df)
    axial_keypoints_df = pd.read_csv(args.axial_keypint_df)
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

    models_stage2, model_names = get_models(
        stage="2",
        exp_id=EXP_ID,
        series_description=None,
        fold=None,
        logdirs=args.stage2_logdirs,
        log_root=args.log_root,
        device=args.device,
    )

    preds_models = defaultdict(dict)

    for study_id in tqdm(series_description_df.study_id.unique()):
        series_df = sagittal_keypoints_df[sagittal_keypoints_df.study_id == study_id]

        series_ids = series_df[
            series_df.series_description == "Sagittal T1"
        ].series_id.unique()
        st1_features = []
        st1_masks = []
        for series_id in series_ids:
            volume = get_volume(
                series_id=series_id,
                description="Sagittal T1",
                dicoms_root=args.dicoms_root,
            )
            keypoints = (
                series_df[series_df.series_id == series_id]
                .sort_values("level")
                .loc[:, ["x", "y"]]
            ).to_numpy()  # (5, 2)
            cropped_volume, mask = preprocess_stage2_sagittal(
                volume,
                keypoints,
                img_size=args.img_size_st1,
                in_chans=args.in_chans_st1,
                crop_range=args.crop_range_st1,
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
                        (5, args.in_chans_st1, model.transformer_dim),
                        device=device,
                        dtype=torch.float32,
                    )
                    for model in models_stage2
                ]
            )
            st1_masks.append(torch.zeros((5, args.in_chans_st1), device=device))

        series_ids = series_df[
            series_df.series_description == "Sagittal T2/STIR"
        ].series_id.unique()
        st2_features = []
        st2_masks = []
        for series_id in series_ids:
            volume = get_volume(
                series_id=series_id,
                description="Sagittal T2/STIR",
                dicoms_root=args.dicoms_root,
            )
            keypoints = (
                series_df[series_df.series_id == series_id]
                .sort_values("level")
                .loc[:, ["x", "y"]]
            ).to_numpy()  # (5, 2)
            cropped_volume, mask = preprocess_stage2_sagittal(
                volume,
                keypoints,
                img_size=args.img_size_st2,
                in_chans=args.in_chans_st2,
                crop_range=args.crop_range_st2,
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
                        (5, args.in_chans_st2, model.transformer_dim),
                        device=device,
                        dtype=torch.float32,
                    )
                    for model in models_stage2
                ]
            )
            st2_masks.append(torch.zeros((5, args.in_chans_st2), device=device))

        axial_series_df = axial_keypoints_df[axial_keypoints_df.study_id == study_id]
        series_ids = axial_series_df.series_id.unique()
        ax_features = []
        ax_masks = []
        for series_id in series_ids:
            volume = get_volume(
                series_id=series_id,
                description="Axial T2",
                dicoms_root=args.dicoms_root,
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
                img_size=args.img_size_ax,
                in_chans=args.in_chans_ax,
                crop_range=args.crop_range_ax,
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
                        (5, args.in_chans_ax * 2, model.transformer_dim),
                        device=device,
                        dtype=torch.float32,
                    )
                    for model in models_stage2
                ]
            )
            ax_masks.append(torch.zeros((5, args.in_chans_ax * 2), device=device))

        for i, model_name in enumerate(model_names):
            preds_models[model_name][study_id] = []
        for st1_feat, st1_mask in zip(st1_features, st1_masks):
            for st2_feat, st2_mask in zip(st2_features, st2_masks):
                for ax_feat, ax_mask in zip(ax_features, ax_masks):
                    for i, (model, st1_f, st2_f, ax_f) in enumerate(
                        zip(models_stage2, st1_feat, st2_feat, ax_feat)
                    ):
                        preds_models[model_names[i]][study_id].append(
                            forward_head_stage2(
                                model,
                                st1_f,
                                st2_f,
                                ax_f,
                                torch.cat([st1_mask, st2_mask, ax_mask], 1).bool(),
                            )
                        )
        for model_name in model_names:
            preds_models[model_name][study_id] = np.stack(
                preds_models[model_name][study_id]
            ).mean(0)
    conditions = [
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "spinal_canal_stenosis",
        "left_subarticular_stenosis",
        "right_subarticular_stenosis",
    ]
    levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
    for model_name, preds_study_id in preds_models.items():
        row_id = []
        pred_values = []
        for logdir in args.stage2_logdirs:
            if logdir in model_name:
                break
        for i in range(5):
            if f"fold{i}" in model_name:
                fold = i
                break
        for study_id in preds_study_id.keys():
            for c in conditions:
                for l in levels:
                    row_id.append(f"{study_id}_{c}_{l}")
            pred_values.append(np.asarray(preds_study_id[study_id]).reshape(25, 3))
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
        preds_df.to_csv("submission_" + logdir + f"_fold{fold}" + ".csv", index=False)


if __name__ == "__main__":
    main(get_args())
