import argparse
from datasets import load_dataset, load_from_disk, concatenate_datasets
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import get_data_from_h5, multi_session_dataset_iTransformer, split_unaligned_dataset
from models.ndt1_v0 import NDT1
from models.ndtgpt import NDTGPT
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

# Fix Args
EID_PATH = 'data/target_eids.txt'

# Dynamic Args
ap = argparse.ArgumentParser()
ap.add_argument("--model_name", type=str, default="NDT-GPT")
ap.add_argument("--eid", type=str, default='671c7ea7-6726-4fbe-adeb-f89c2c8e489b')
ap.add_argument("--base_path", type=str, default='/expanse/lustre/scratch/zwang34/temp_project/random_exp')
ap.add_argument("--train", action='store_true')
ap.add_argument("--eval", action='store_true')
ap.add_argument("--probe", action='store_true')
ap.add_argument("--overwrite", action='store_true')  # TODO: implement this
ap.add_argument("--unaligned_training", action='store_true')
ap.add_argument("--epochs", type=int, default=1000)
ap.add_argument("--suffix", type=str, default='common')
args = ap.parse_args()
eid = args.eid

# load config
kwargs = {
    "model": "include:src/configs/ndtgpt/ndtgpt.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndtgpt/trainer_ndtgpt.yaml", config)
config = update_config("src/configs/ndtgpt/probe.yaml", config)

# Update config by dynamic args
config['training']['num_epochs'] = args.epochs

# wandb
prefix = ''
if args.train:
    prefix += 'T'
if args.eval:
    prefix += 'E'
if args.probe:
    prefix += 'P'

if config.wandb.use:
    import wandb
    wandb.init(
        project=config.wandb.project, 
        entity=config.wandb.entity, 
        config=config,
        name="({}){}_model_{}_method_{}_mask_{}_ratio_{}_ual_training_{}_{}".format(
            prefix,
            eid[:5],
            config.model.model_class, config.method.model_kwargs.method_name, 
            args.mask_mode, args.mask_ratio, args.unaligned_training, args.suffix,
        )
    )

last_ckpt_path = 'last' 
best_ckpt_path = 'best' 

# ------------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------------

# make log dir
log_dir = os.path.join(
    args.base_path, 
    eid, 
    "train", 
    "model_{}".format(config.model.model_class),
    "method_{}".format(config.method.model_kwargs.method_name), 
    "mask_{}".format(args.mask_mode),
    "ratio_{}".format(args.mask_ratio),
    "ual_training_{}".format(args.unaligned_training),
    args.suffix,
)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# showing current configuration
print("Training log directory: {}".format(log_dir))

if args.train:
    
    print("=================================")
    print("            Training             ")
    print("=================================")   
    
    # set seed for reproducibility
    set_seed(config.seed)
    
    # download dataset from huggingface
    if args.unaligned_training:
        _al = load_dataset(f'neurofm123/{eid}_aligned', cache_dir=config.dirs.dataset_cache_dir, download_mode='force_redownload')
        _ual = load_dataset(f'neurofm123/{eid}', cache_dir=config.dirs.dataset_cache_dir, download_mode='force_redownload')
        dataset = split_unaligned_dataset(_al, _ual)
        train_dataset = dataset["train"]
        val_dataset = dataset["val"]
        test_dataset = dataset["test"]
    else:
        dataset = load_dataset(f'neurofm123/{eid}_aligned', cache_dir=config.dirs.dataset_cache_dir, download_mode='force_redownload')
        train_dataset = dataset["train"]
        val_dataset = dataset["val"]
        test_dataset = dataset["test"]
    try:
        bin_size = train_dataset["binsize"][0]
    except:
        bin_size = train_dataset["bin_size"][0]
    print(train_dataset.column_names)
    print(f"bin_size: {bin_size}")

    # update the num_neurons related quantity
    num_neurons = len(train_dataset[0]['cluster_uuids'])
    config['model']['encoder']['embedder']['n_channels'] = num_neurons
    config['data']['max_space_length'] = num_neurons
    print(f'number of neurons: {num_neurons}')
    
    # make the dataloader
    train_dataloader = make_loader(train_dataset, 
                             target=config.data.target,
                             load_meta=config.data.load_meta,
                             batch_size=config.training.train_batch_size, 
                             pad_to_right=True, 
                             pad_value=-1.,
                             bin_size=bin_size,
                             max_time_length=config.data.max_time_length,
                             max_space_length=config.data.max_space_length,
                             dataset_name=config.data.dataset_name,
                             sort_by_depth=config.data.sort_by_depth,
                             sort_by_region=config.data.sort_by_region,
                             shuffle=True)
    
    val_dataloader = make_loader(val_dataset, 
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
    
    # Initialize the accelerator
    accelerator = Accelerator()
    
    # load model
    NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch, "iTransformer": iTransformer, "NDT-GPT": NDTGPT}
    model_class = NAME2MODEL[config.model.model_class]
    model = model_class(config.model, **config.method.model_kwargs)
    model = accelerator.prepare(model)
    
    # print all the configurations, and the model itself
    print('--------------------------------------------------')
    print('Training Configuration')
    print('--------------------------------------------------')
    print(model)
    print(model.config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)
    lr_scheduler = OneCycleLR(
                    optimizer=optimizer,
                    total_steps=config.training.num_epochs*len(train_dataloader) //config.optimizer.gradient_accumulation_steps,
                    max_lr=config.optimizer.lr,
                    pct_start=config.optimizer.warmup_pct,
                    div_factor=config.optimizer.div_factor,
                )
    
    trainer_kwargs = {
        "log_dir": log_dir,
        "accelerator": accelerator,
        "lr_scheduler": lr_scheduler,
        "config": config,
    }
    trainer_ = make_trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        **trainer_kwargs
    )
    
    # train loop
    trainer_.train()

# ------------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------------
if args.eval:
    
    print("=================================")
    print("           Evaluation            ")
    print("=================================")

    raise NotImplementedError("Evaluation is not implemented yet.")

# behavior probe
if args.probe:

    print("=================================")
    print("             Probe               ")
    print("=================================")
    
    '''probe_configs = {
        'model_config': 'src/configs/ndt1_v0/ndt1_v0.yaml',
        'trainer_config': "src/configs/ndt1_v0/trainer_ndt1_v0.yaml",
        'probe_config': "src/configs/ndt1_v0/probe.yaml",
        'model_path': os.path.join(log_dir, best_ckpt_path),
        'mask_mode': args.mask_mode,
        'seed': config.seed,
        'eid': eid,
    }

    behavior_probe_eval(**probe_configs)'''

    raise NotImplementedError("Probe is not implemented yet.")

            
    