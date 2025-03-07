# python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name UNet --val_freq 5
# python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name UNet --val_freq 5
# python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name UNet --val_freq 5
# python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name UNet --val_freq 5

# python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name MambaUNet --val_freq 10
# python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name MambaUNet --val_freq 10
# python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name MambaUNet --val_freq 10
# python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name MambaUNet --val_freq 10

python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 10 --use_down_sample False
python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 10 --use_down_sample False
python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 10 --use_down_sample False
python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 10 --use_down_sample False

python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 10
python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 10
python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 10
python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name SpatialMambaHSI --val_freq 10

# python train.py --dataset_index 0 --data_set_path /public/MambaHSI-DATA --net_name MambaHSI
# python train.py --dataset_index 1 --data_set_path /public/MambaHSI-DATA --net_name MambaHSI
# python train.py --dataset_index 2 --data_set_path /public/MambaHSI-DATA --net_name MambaHSI
# python train.py --dataset_index 3 --data_set_path /public/MambaHSI-DATA --net_name MambaHSI