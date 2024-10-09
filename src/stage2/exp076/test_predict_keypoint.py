import argparse
import os
import sys
from glob import glob

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
from albumentations.pytorch import ToTensorV2
from evaluation import (
    get_models,
    get_volume,
    prediction_stage1_sagittal,
    preprocess_stage1_sagittal,
)
from torch.nn import functional as F
from tqdm.auto import tqdm


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=True)
    parent_parser.add_argument("--log_root", default="../../../logs")
    parent_parser.add_argument("--fold", nargs="+", default=[0, 1, 2, 3, 4], type=int)
    parent_parser.add_argument(
        "--stage1_exp_id",
        default="017",
    )
    parent_parser.add_argument(
        "--stage1_logdirs",
        nargs="+",
        default=[
            "caformer_s18_unet_scse_20x256x256_mixup",
            "convnext_tiny_unet_scse_20x256x256_mixup",
            "resnetrs50_unet_scse_20x256x256_mixup",
            "swinv2_tiny_unet_scse_20x256x256_mixup",
        ],
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
        "--out",
        default="sagittal_keypoints_df.csv",
    )
    return parent_parser.parse_args()


def main(args):
    series_description_df = pd.read_csv(args.series_description_df)
    device = args.device
    sagittal_t1_result = []
    sagittal_t2_result = []

    model_stage1_sagittal_t1, _ = get_models(
        stage="1",
        exp_id=args.stage1_exp_id,
        series_description="Sagittal_T1",
        fold=None,
        logdirs=args.stage1_logdirs,
        log_root=args.log_root,
        device=device,
    )
    for series_id in tqdm(
        series_description_df[
            series_description_df.series_description == "Sagittal T1"
        ].series_id.unique()
    ):
        volume = get_volume(
            series_id=series_id, description="Sagittal T1", dicoms_root=args.dicoms_root
        )
        stage1_volume = preprocess_stage1_sagittal(volume).to(device)
        keypoints = prediction_stage1_sagittal(model_stage1_sagittal_t1, stage1_volume)
        for i, level in enumerate((["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"])):
            sagittal_t1_result.append(
                [series_id, level, keypoints[i][0], keypoints[i][1]]
            )
        # break
    del model_stage1_sagittal_t1
    torch.cuda.empty_cache()

    model_stage1_sagittal_t2, _ = get_models(
        stage="1",
        exp_id=args.stage1_exp_id,
        series_description="Sagittal_T2-STIR",
        fold=None,
        logdirs=args.stage1_logdirs,
        log_root=args.log_root,
        device=device,
    )
    for series_id in tqdm(
        series_description_df[
            series_description_df.series_description == "Sagittal T2/STIR"
        ].series_id.unique()
    ):
        volume = get_volume(
            series_id=series_id,
            description="Sagittal T2/STIR",
            dicoms_root=args.dicoms_root,
        )
        stage1_volume = preprocess_stage1_sagittal(volume).to(device)
        keypoints = prediction_stage1_sagittal(model_stage1_sagittal_t2, stage1_volume)
        for i, level in enumerate((["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"])):
            sagittal_t2_result.append(
                [series_id, level, keypoints[i][0], keypoints[i][1]]
            )
    del model_stage1_sagittal_t2
    torch.cuda.empty_cache()

    sagittal_keypoints_df = pd.DataFrame(
        sagittal_t1_result + sagittal_t2_result,
        columns=["series_id", "level", "x", "y"],
    ).reset_index(drop=True)
    sagittal_keypoints_df = sagittal_keypoints_df.merge(
        series_description_df.loc[:, ["series_id", "series_description", "study_id"]],
        on=["series_id"],
    )
    sagittal_keypoints_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main(get_args())
