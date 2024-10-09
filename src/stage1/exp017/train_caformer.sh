python train.py --seed 2034 --logdir caformer_s18_unet_scse_20x256x256_mixup --fold 0 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name caformer_s18.sail_in22k_ft_in1k_384 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2035 --logdir caformer_s18_unet_scse_20x256x256_mixup --fold 1 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name caformer_s18.sail_in22k_ft_in1k_384 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2036 --logdir caformer_s18_unet_scse_20x256x256_mixup --fold 2 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name caformer_s18.sail_in22k_ft_in1k_384 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2039 --logdir caformer_s18_unet_scse_20x256x256_mixup --fold 3 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name caformer_s18.sail_in22k_ft_in1k_384 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 2038 --logdir caformer_s18_unet_scse_20x256x256_mixup --fold 4 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T1 --model_name caformer_s18.sail_in22k_ft_in1k_384 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16


python train.py --seed 3034 --logdir caformer_s18_unet_scse_20x256x256_mixup --fold 0 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name caformer_s18.sail_in22k_ft_in1k_384 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3035 --logdir caformer_s18_unet_scse_20x256x256_mixup --fold 1 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name caformer_s18.sail_in22k_ft_in1k_384 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3036 --logdir caformer_s18_unet_scse_20x256x256_mixup --fold 2 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name caformer_s18.sail_in22k_ft_in1k_384 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3037 --logdir caformer_s18_unet_scse_20x256x256_mixup --fold 3 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name caformer_s18.sail_in22k_ft_in1k_384 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

python train.py --seed 3038 --logdir caformer_s18_unet_scse_20x256x256_mixup --fold 4 --gpus 4 --epochs 50 \
    --precision 16-mixed --series_description Sagittal_T2-STIR --model_name caformer_s18.sail_in22k_ft_in1k_384 --attention_type scse \
    --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --num_workers 6 --batch_size 16

