import argparse
from datasets import load_dataset, load_from_disk, concatenate_datasets
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import get_data_from_h5, multi_session_dataset_iTransformer, split_unaligned_dataset
from models.ndt1_v0 import NDT1
from models.stpatch import STPatch
from models.itransformer_multi import iTransformer
from torch.optim.lr_scheduler import OneCycleLR
import torch
import numpy as np
import os
from trainer.make import make_trainer
from utils.eval_utils import load_model_data_local, co_smoothing_eval, behavior_probe_eval
import warnings
warnings.simplefilter("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="671c7ea7-6726-4fbe-adeb-f89c2c8e489b")
args = ap.parse_args()
eid = args.eid

# choose the target idxs.
_idxs = np.array([49, 222,  50, 177, 254, 113,  81,  62, 130,  60,  38, 234])
target_idxs = np.zeros(num_neurons, dtype=bool)
target_idxs[_idxs] = True


# randomly choose some configs to use here. doesn't matter.
# load config
kwargs = {
    "model": "include:src/configs/ndt1_v0/ndt1_v0.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndt1_v0/trainer_ndt1_v0.yaml", config)
config = update_config("src/configs/ndt1_v0/probe.yaml", config)

# load dataset
dataset = load_dataset(f'ibl-foundation-model/{eid}_aligned', cache_dir=config.dirs.dataset_cache_dir, download_mode='force_redownload')
train_dataset = dataset["train"]
val_dataset = dataset["val"]
test_dataset = dataset["test"]
try:
    bin_size = train_dataset["binsize"][0]
except:
    bin_size = train_dataset["bin_size"][0]
print(train_dataset.column_names)
print(f"bin_size: {bin_size}")

num_neurons = len(train_dataset[0]['cluster_uuids'])
config['data']['max_space_length'] = num_neurons
print(f'number of neurons: {num_neurons}')

# make the dataloader. (only test set is needed during smoothing evaluation)
test_dataloader = make_loader(test_dataset, 
                         target=config.data.target,
                         load_meta=config.data.load_meta,
                         batch_size=config.training.test_batch_size, 
                         pad_to_right=True, 
                         pad_value=-1.,
                         bin_size=bin_size,
                         max_time_length=config.data.max_time_length,
                         max_space_length=config.data.max_space_length,
                         dataset_name=config.data.dataset_name,
                         sort_by_depth=config.data.sort_by_depth,
                         sort_by_region=config.data.sort_by_region,
                         shuffle=False)

bps_result_list, r2_result_list = [float('nan')] * num_neurons, [np.array([np.nan, np.nan])] * num_neurons

gt_list = []
pred_list = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = move_batch_to_device(batch, accelerator.device)
        gt = batch['spikes_data'][:, :, target_idxs].clone()
        gt_list.append(gt)













