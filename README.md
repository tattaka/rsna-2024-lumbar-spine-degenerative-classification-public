# 4th place solution (tattaka's part)

## Requirements
16GB x 4 VRAM (trained on NVIDIA RTX A4000 x 4).

## Environment
Use Kaggle Docker.  
Follow tattaka/ml_environment to build the environment.  
You can run ./RUN-KAGGLE-GPU-ENV.sh and launch the docker container.  

## Usage
0. Place [competition data](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data) in the `input` directory
1. Place [Lumbar Coordinate Dataset](https://www.kaggle.com/datasets/brendanartley/lumbar-coordinate-pretraining-dataset) in the `input` directory
2. Run the following script for keypoint detection.
   1. `cd input && python python save_volume_256x256.py`
   2. `cd ../src/stage1 && sh train_caformer.sh && sh train_convnext.sh && sh train_resnetrs50.sh && sh train_swinv2.sh`
3. Run the following script for classification.
   1. `cd input && python python save_volume_orig_res.py`
   2. Run all cell of `make_level_df.ipynb`
   3. `cd ../src/stage2/exp076` and run all cell of `predict_keypoint.ipynb` and `make_train_df.ipynb`
   4. `sh train_caformer_s18.sh && sh train_resnetrs50.sh && sh train_swinv2_tiny.sh`
   5. The same procedure is applied for exp107 as well.

## License
MIT
