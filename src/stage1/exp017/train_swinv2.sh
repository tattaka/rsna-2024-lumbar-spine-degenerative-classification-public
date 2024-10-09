python train.py --seed 2044 --logdir swinv2_tiny_unet_scse_20x256x256_mixup --fold 0 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name swinv2_tiny_window8_256.ms_in1k --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2045 --logdir swinv2_tiny_unet_scse_20x256x256_mixup --fold 1 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name swinv2_tiny_window8_256.ms_in1k --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2046 --logdir swinv2_tiny_unet_scse_20x256x256_mixup --fold 2 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name swinv2_tiny_window8_256.ms_in1k --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2047 --logdir swinv2_tiny_unet_scse_20x256x256_mixup --fold 3 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name swinv2_tiny_window8_256.ms_in1k --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2048 --logdir swinv2_tiny_unet_scse_20x256x256_mixup --fold 4 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name swinv2_tiny_window8_256.ms_in1k --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.1 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16


python train.py --seed 3044 --logdir swinv2_tiny_unet_scse_20x256x256_mixup --fold 0 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name swinv2_tiny_window8_256.ms_in1k --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3045 --logdir swinv2_tiny_unet_scse_20x256x256_mixup --fold 1 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name swinv2_tiny_window8_256.ms_in1k --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3046 --logdir swinv2_tiny_unet_scse_20x256x256_mixup --fold 2 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name swinv2_tiny_window8_256.ms_in1k --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3047 --logdir swinv2_tiny_unet_scse_20x256x256_mixup --fold 3 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name swinv2_tiny_window8_256.ms_in1k --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3048 --logdir swinv2_tiny_unet_scse_20x256x256_mixup --fold 4 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name swinv2_tiny_window8_256.ms_in1k --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

