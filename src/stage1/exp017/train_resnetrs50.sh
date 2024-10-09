python train.py --seed 2024 --logdir resnetrs50_unet_scse_20x256x256_mixup --fold 0 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name resnetrs50 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2025 --logdir resnetrs50_unet_scse_20x256x256_mixup --fold 1 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name resnetrs50 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2026 --logdir resnetrs50_unet_scse_20x256x256_mixup --fold 2 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name resnetrs50 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2027 --logdir resnetrs50_unet_scse_20x256x256_mixup --fold 3 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name resnetrs50 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2028 --logdir resnetrs50_unet_scse_20x256x256_mixup --fold 4 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name resnetrs50 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16


python train.py --seed 3024 --logdir resnetrs50_unet_scse_20x256x256_mixup --fold 0 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name resnetrs50 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3025 --logdir resnetrs50_unet_scse_20x256x256_mixup --fold 1 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name resnetrs50 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3026 --logdir resnetrs50_unet_scse_20x256x256_mixup --fold 2 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name resnetrs50 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3027 --logdir resnetrs50_unet_scse_20x256x256_mixup --fold 3 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name resnetrs50 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3028 --logdir resnetrs50_unet_scse_20x256x256_mixup --fold 4 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name resnetrs50 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16
