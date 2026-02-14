# Baseline MambaHSI

# python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name MambaHSI --val_freq 50 --train_samples 80
# python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name MambaHSI --val_freq 50 # --train_samples 30
# python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name MambaHSI --val_freq 50 # --train_samples 15
# python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name MambaHSI --val_freq 50 # --train_samples 15

# UNet variants

# python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name UNet --val_freq 50
# python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name UNet --val_freq 50
# python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name UNet --val_freq 50
# python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name UNet --val_freq 50
# python train.py --dataset_index 4 --data_set_path /public/MambaHSI-DATA --net_name UNet --val_freq 50

# MambaUNetHSI variants

# python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name MambaUNetHSI --val_freq 20 --scale_num 2
# python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name MambaUNetHSI --val_freq 20 --scale_num 2
# python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name MambaUNetHSI --val_freq 20 --scale_num 2
# python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name MambaUNetHSI --val_freq 20 --scale_num 2

# SpatialMambaHSI variants

python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 1
python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 1
python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 1
python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 1
python train.py --dataset_index 4 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 1

python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 2
python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 2
python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 2
python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 2
python train.py --dataset_index 4 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 2

python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 3
python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 3
python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 3
python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 3
python train.py --dataset_index 4 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 3

# MambaVisionHSI variants

# python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 1
# python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 1
# python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 1
# python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 1
# python train.py --dataset_index 4 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 1

# python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 2
# python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 2
# python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 2
# python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 2
# python train.py --dataset_index 4 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 2

# python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 3
# python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 3
# python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 3
# python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name MambaVisionHSI --val_freq 50 --use_down_sample --hidden_dim 128 --depth 3
