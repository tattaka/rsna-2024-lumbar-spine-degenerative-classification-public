from glob import glob

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
from IPython import display
from matplotlib import animation
from torch.nn import functional as F
from tqdm.auto import trange

DATA_PATH = "rsna-2024-lumbar-spine-degenerative-classification/"
train_descriptions = pd.read_csv(DATA_PATH + "train_series_descriptions.csv")


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


def get_volume(description_df: pd.DataFrame, idx: int):
    series_id = description_df.iloc[idx].series_id
    series_description = description_df.iloc[idx].series_description
    volume = []
    img_paths = glob(DATA_PATH + f"train_images/**/{series_id}/*.dcm")
    isid = sorted([int(p.split("/")[-1].split(".")[0]) for p in img_paths])
    for i in isid:
        img_path = glob(
            DATA_PATH + f"train_images/**/{series_id}/{i}.dcm",
            recursive=True,
        )[0]
        dicom = pydicom.dcmread(img_path)
        img = standardize_pixel_array(dicom)
        # img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        volume.append(
            F.interpolate(
                torch.Tensor(img)[None, None, ...],
                size=(256, 256),
                align_corners=True,
                mode="bilinear",
            ).numpy()[0, 0]
        )
    volume = np.stack(volume)[None, None, ...]
    if series_description == "Axial T2":
        volume = F.interpolate(
            torch.Tensor(volume),
            size=(50, 256, 256),
            mode="trilinear",
            align_corners=True,
        ).numpy()[0, 0]
    else:
        volume = F.interpolate(
            torch.Tensor(volume),
            size=(20, 256, 256),
            mode="trilinear",
            align_corners=True,
        ).numpy()[0, 0]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
    volume = (volume * 255).astype(np.uint8)
    return volume, series_id


h5f = h5py.File("resize_volume_256x256.h5", "a")
for idx in trange(len(train_descriptions)):
    volume, series_id = get_volume(train_descriptions, idx)
    h5f.create_dataset(f"{series_id}", data=volume)
h5f.close()
