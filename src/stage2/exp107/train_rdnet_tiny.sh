python train.py --seed 6124 --logdir rdnet_tiny_30x1x128x128_x1.0_30x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1 --fold 0 \
    --precision 16-mixed --gpus 4 --epochs 50 --lr 2e-4 \
    --backbone_lr 2e-4 \
    --num_workers 4 --batch_size 4 --mixup_p 0.5 --mixup_alpha 0.5 \
    --crop_range_st1 1.0 --in_chans_st1 30 --img_size_st1 128 \
    --crop_range_st2 1.0 --in_chans_st2 30 --img_size_st2 128 \
    --crop_range_ax 1.0 --in_chans_ax 5 --img_size_ax 128 \
    --model_name rdnet_tiny.nv_in1k --drop_path_rate 0.2 \
    --transformer_dim 256 --transformer_num_layers 2 --transformer_nhead 8 --max_token_mask_rate 0.1

python train.py --seed 6125 --logdir rdnet_tiny_30x1x128x128_x1.0_30x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1 --fold 1 \
    --precision 16-mixed --gpus 4 --epochs 50 --lr 2e-4 \
    --backbone_lr 2e-4 \
    --num_workers 4 --batch_size 4 --mixup_p 0.5 --mixup_alpha 0.5 \
    --crop_range_st1 1.0 --in_chans_st1 30 --img_size_st1 128 \
    --crop_range_st2 1.0 --in_chans_st2 30 --img_size_st2 128 \
    --crop_range_ax 1.0 --in_chans_ax 5 --img_size_ax 128 \
    --model_name rdnet_tiny.nv_in1k --drop_path_rate 0.2 \
    --transformer_dim 256 --transformer_num_layers 2 --transformer_nhead 8 --max_token_mask_rate 0.1

python train.py --seed 6126 --logdir rdnet_tiny_30x1x128x128_x1.0_30x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1 --fold 2 \
    --precision 16-mixed --gpus 4 --epochs 50 --lr 2e-4 \
    --backbone_lr 2e-4 \
    --num_workers 4 --batch_size 4 --mixup_p 0.5 --mixup_alpha 0.5 \
    --crop_range_st1 1.0 --in_chans_st1 30 --img_size_st1 128 \
    --crop_range_st2 1.0 --in_chans_st2 30 --img_size_st2 128 \
    --crop_range_ax 1.0 --in_chans_ax 5 --img_size_ax 128 \
    --model_name rdnet_tiny.nv_in1k --drop_path_rate 0.2 \
    --transformer_dim 256 --transformer_num_layers 2 --transformer_nhead 8 --max_token_mask_rate 0.1 

python train.py --seed 6127 --logdir rdnet_tiny_30x1x128x128_x1.0_30x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1 --fold 3 \
    --precision 16-mixed --gpus 4 --epochs 50 --lr 2e-4 \
    --backbone_lr 2e-4 \
    --num_workers 4 --batch_size 4 --mixup_p 0.5 --mixup_alpha 0.5 \
    --crop_range_st1 1.0 --in_chans_st1 30 --img_size_st1 128 \
    --crop_range_st2 1.0 --in_chans_st2 30 --img_size_st2 128 \
    --crop_range_ax 1.0 --in_chans_ax 5 --img_size_ax 128 \
    --model_name rdnet_tiny.nv_in1k --drop_path_rate 0.2 \
    --transformer_dim 256 --transformer_num_layers 2 --transformer_nhead 8 --max_token_mask_rate 0.1 

python train.py --seed 6128 --logdir rdnet_tiny_30x1x128x128_x1.0_30x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1 --fold 4 \
    --precision 16-mixed --gpus 4 --epochs 50 --lr 2e-4 \
    --backbone_lr 2e-4 \
    --num_workers 4 --batch_size 4 --mixup_p 0.5 --mixup_alpha 0.5 \
    --crop_range_st1 1.0 --in_chans_st1 30 --img_size_st1 128 \
    --crop_range_st2 1.0 --in_chans_st2 30 --img_size_st2 128 \
    --crop_range_ax 1.0 --in_chans_ax 5 --img_size_ax 128 \
    --model_name rdnet_tiny.nv_in1k --drop_path_rate 0.2 \
    --transformer_dim 256 --transformer_num_layers 2 --transformer_nhead 8 --max_token_mask_rate 0.1 

python evaluation.py --stage2_logdirs rdnet_tiny_30x1x128x128_x1.0_30x1x128x128_x1.0_5x1x128x128_x1.0_mixup_mask0.1 \
    --crop_range_st1 1.0 --in_chans_st1 30 --img_size_st1 128 \
    --crop_range_st2 1.0 --in_chans_st2 30 --img_size_st2 128 \
    --crop_range_ax 1.0 --in_chans_ax 5 --img_size_ax 128 \
    --out eval_rdnet_tiny_ax5ch_mask0.1.csv