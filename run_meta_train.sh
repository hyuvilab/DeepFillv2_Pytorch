python3 train.py \
--baseroot '/home/dataset/places2/small_image/data_256' \
--save_path './models' \
--sample_path './samples' \
--gpu_ids '0' \
--gan_type 'WGAN' \
--cudnn_benchmark True \
--checkpoint_interval 1 \
--multi_gpu True \
--load_name '' \
--epochs 20 \
--batch_size 16 \
--lr_g 1e-4 \
--lr_d 1e-4 \
--lambda_l1 10 \
--lambda_perceptual 10 \
--lambda_gan 1 \
--lr_decrease_epoch 10 \
--lr_decrease_factor 0.5 \
--num_workers 6 \
--in_channels 4 \
--out_channels 3 \
--latent_channels 48 \
--pad_type 'zero' \
--activation 'elu' \
--norm 'none' \
--init_type 'kaiming' \
--init_gain 0.02 \
--imgsize 256 \
--mask_type 'free_form' \
--margin 10 \
--mask_num 20 \
--bbox_shape 30 \
--max_angle 4 \
--max_len 20 \
--max_width 5 \
--logs_dir_path '/home/eh1404/works/DeepFillv2_Pytorch/models/logs' \
--meta_training