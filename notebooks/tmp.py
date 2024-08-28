# Path
import os, sys
os.chdir('/home/zwang34/IBL/iblfm_exp/IBL_foundation_model')
sys.path.append('./src')
print(sys.path)

import logging
logging.getLogger().setLevel(logging.ERROR)

# Lib
from datasets import load_dataset, concatenate_datasets
import numpy as np
from loader.make_loader import make_loader
from utils.eval_utils import bits_per_spike
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from accelerate import Accelerator
from src.loader.make_loader import make_loader
from src.utils.dataset_utils import split_both_dataset, multi_session_zs_dataset_iTransformer, multi_session_dataset_iTransformer
from src.utils.utils import set_seed, move_batch_to_device, plot_gt_pred, metrics_list, plot_avg_rate_and_spike, \
    plot_rate_and_spike
from src.utils.config_utils import config_from_kwargs, update_config
from src.utils.hooks_utils import HookManager
from src.models.ndt1_v0 import NDT1  # only here is not compatible. Need to change this for non-regression.
from models.ndtgpt import NDTGPT
from src.models.stpatch import STPatch
from src.models.itransformer_multi import iTransformer  # use multi-version for now
from src.models.probe_decoders import ProbeDecoder
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import r2_score
from scipy.special import gammaln
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.colors as colors
import os
import re
from src.trainer.make import make_trainer
from pathlib import Path

# Fix Args
EID = '671c7ea7-6726-4fbe-adeb-f89c2c8e489b'
kernel_sigma_list = [1, 2, 3, 8, 16, 32]
fr_filter = 0.1 # unit: 1 Hz

## Prepare data
dataset = load_dataset(f'neurofm123/{EID}_aligned', cache_dir='/expanse/lustre/scratch/zwang34/temp_project/iTransformer/checkpoints/datasets_cache')
train_dataset = dataset['train']
val_dataset = dataset['val']
test_dataset = dataset['test']
n_neurons = len(test_dataset[0]['cluster_uuids'])

train_dataloader = make_loader(
    train_dataset,
    target=None,
    load_meta=True,
    batch_size=10000,
    pad_to_right=True,
    pad_value=-1,
    bin_size=0.02,
    max_time_length=100,
    max_space_length=n_neurons,
    dataset_name='ibl',
    shuffle=True,
)

val_dataloader = make_loader(
    val_dataset,
    target=None,
    load_meta=True,
    batch_size=10000,
    pad_to_right=True,
    pad_value=-1,
    bin_size=0.02,
    max_time_length=100,
    max_space_length=n_neurons,
    dataset_name='ibl',
    shuffle=False,
)

test_dataloader = make_loader(
    test_dataset,
    target=None,
    load_meta=True,
    batch_size=10000,
    pad_to_right=True,
    pad_value=-1,
    bin_size=0.02,
    max_time_length=100,
    max_space_length=n_neurons,
    dataset_name='ibl',
    shuffle=False,
)

for batch in train_dataloader:
    train_data = batch['spikes_data'].detach().cpu().numpy()
for batch in val_dataloader:
    val_data = batch['spikes_data'].detach().cpu().numpy()
for batch in test_dataloader:
    test_data = batch['spikes_data'].detach().cpu().numpy()

# Use a fr Filter
whole_data = np.concatenate([train_data, val_data, test_data], axis=0)
mean_fr = np.mean(whole_data, axis=(0,1)) * 50   # hz
valid_idx = (mean_fr >= fr_filter)
train_data = train_data[:, :, valid_idx]
val_data = val_data[:, :, valid_idx]
test_data = test_data[:, :, valid_idx]
print(f'valid neuron: {sum(valid_idx)}, invalid neuron: {len(valid_idx)-sum(valid_idx)}')

for k, kernel_sigma in enumerate(kernel_sigma_list):       
    gt_spikes = test_data
    smoothed_spikes = gaussian_filter1d(gt_spikes, sigma=kernel_sigma, axis=1)
    fr_stat = np.log(np.mean(gt_spikes, axis=(0, 1))+1e-9)
    
    bps_smth_list = []
    bps_gt_list = []
    for i in range(sum(valid_idx)):
        bps_smth_list.append(bits_per_spike(smoothed_spikes[:, :, [i]], gt_spikes[:, :, [i]]))
        bps_gt_list.append(bits_per_spike(gt_spikes[:, :, [i]], gt_spikes[:, :, [i]]))
    population_smth_bps = bits_per_spike(smoothed_spikes, gt_spikes)
    population_upb_bps = bits_per_spike(gt_spikes, gt_spikes)

    mean_smth_bps = np.mean(bps_smth_list)
    mean_upb_bps = np.mean(bps_gt_list)
    print(f'{kernel_sigma*20}ms smoothing bps: {mean_smth_bps}, gt bps: {mean_upb_bps}. Population bps: {population_smth_bps}, {population_upb_bps}')

    
    '''
    scatter = axes.flat[k].scatter(bps_smth_list, bps_gt_list, c=fr_stat)
    axes.flat[k].set_title(f'kernel sigma={kernel_sigma*20}ms')
    axes.flat[k].set_xlim([-10, 15])
    axes.flat[k].set_ylim([-10, 15])
    axes.flat[k].plot([-10, 15], [-10, 15], c='grey')
    '''

'''
axes.flat[0].set_ylabel('gt vs. gt bps')
axes.flat[3].set_ylabel('gt vs. gt bps')
axes.flat[3].set_xlabel('gt vs. smoothed bps')
axes.flat[4].set_xlabel('gt vs. smoothed bps')
axes.flat[5].set_xlabel('gt vs. smoothed bps')
fig.colorbar(scatter)
plt.savefig('smoothing_direct.png')
'''

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

# load config
kwargs = {
    "model": "include:src/configs/ndt1_v0/ndt1_v0.yaml"
}
config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndt1_v0/trainer_ndt1_v0.yaml", config)
config = update_config("src/configs/ndt1_v0/probe.yaml", config)

# Update config by dynamic args
config['model']['encoder']['masker']['mode'] = 'neuron'
config['model']['encoder']['masker']['ratio'] = 0.3
config['training']['num_epochs'] = 100

# make log dir
log_dir = os.path.join(
    '/expanse/lustre/scratch/zwang34/temp_project/random_exp', 
    EID, 
    "train", 
    "model_{}".format(config.model.model_class),
    "method_{}".format(config.method.model_kwargs.method_name), 
    "mask_{}".format('neuron'),
    "ratio_{}".format(0.3),
    "ual_training_{}".format('False'),
    'pos_true',
)

configs = {
    'model_config': 'src/configs/ndt1_v0/ndt1_v0.yaml',
    'model_path': os.path.join(log_dir, 'best'),
    'trainer_config': 'src/configs/ndt1_v0/trainer_ndt1_v0.yaml',
    'seed': config.seed,
    'mask_mode': 'neuron',
    'eid': EID
}

model, accelerator, dataset, dataloader = load_model_data_local(**configs)
num_neurons = len(dataset[0]['cluster_uuids'])

gt_list = []
pred_list = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = move_batch_to_device(batch, accelerator.device)
        gt_list.append(batch['spikes_data'].clone())
        outputs = model(
            batch['spikes_data'].clone(),
            time_attn_mask=batch['time_attn_mask'],
            space_attn_mask=batch['space_attn_mask'],
            spikes_timestamps=batch['spikes_timestamps'],
            spikes_spacestamps=batch['spikes_spacestamps'],
            targets=batch['target'],
            neuron_regions=batch['neuron_regions']
        )
        pred_list.append(outputs.preds)

    gt_spike_data = torch.cat(gt_list, 0)
    pred = torch.cat(pred_list, 0)

    # TODO: fix this for not-poisson condition.
    pred = torch.exp(pred)

    gt_spikes = gt_spike_data.detach().cpu().numpy()
    pred_spikes = pred.detach().cpu().numpy()

    gt_spikes = gt_spikes[:, :, valid_idx]
    pred_spikes = pred_spikes[:, :, valid_idx]
    print(f'valid: {sum(valid_idx)}')

    # population bps
    ndt1_population_bps = bits_per_spike(pred_spikes, gt_spikes)
    print(f'population bps: {ndt1_population_bps}')

    # neuron_average bps
    bps_list = []
    for i in range(gt_spikes.shape[-1]):
        cur_bps = bits_per_spike(pred_spikes[:, :, [i]], gt_spikes[:, :, [i]])
        bps_list.append(cur_bps)
    bps_list = np.array(bps_list)
    bps_list = bps_list[np.isfinite(bps_list)]
    print(f'average bps: {np.mean(bps_list)}')