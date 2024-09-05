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
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.eval_utils import bits_per_spike
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import argparse
import torch
import torch.nn as nn
from src.utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2
import wandb
from models.DeTime import DeTime

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=str, default="linear")
ap.add_argument("--eid", type=str, default="03d9a098-07bf-4765-88b7-85f8d8f620cc")
ap.add_argument("--nonrandomized", action='store_true')
ap.add_argument("--t_res", type=int, default=10)  # unit: 1s
ap.add_argument("--hidden_size", type=int, default=1024)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--eps", type=float, default=1e-8)
ap.add_argument("--epochs", type=int, default=200)
args = ap.parse_args()
EID = args.eid

# wandb
wandb.init(
    project='Detime_Baseline',
    entity='wdk0082',
    config=args.__dict__,
    name=f'{args.model}_{args.eid}_nonrand_{args.nonrandomized}',
)

base_path = '/expanse/lustre/scratch/zwang34/temp_project/random_exp/detime'

save_path = os.path.join(
    base_path,
    args.eid,
    'model_{}_tres_{}s_nonrand_{}'.format(args.model, args.t_res, args.nonrandomized),
)
    

if not os.path.exists(save_path):
    os.makedirs(save_path)


if args.nonrandomized:
    dataset = load_dataset(f'neurofm123/{EID}_nonrandomized', cache_dir='/expanse/lustre/scratch/zwang34/temp_project/iTransformer/checkpoints/datasets_cache', download_mode='force_redownload')
else:
    dataset = load_dataset(f'neurofm123/{EID}_aligned', cache_dir='/expanse/lustre/scratch/zwang34/temp_project/iTransformer/checkpoints/datasets_cache', download_mode='force_redownload')

train_dataset = dataset['train']
val_dataset = dataset['val']
test_dataset = dataset['test']
n_neurons = len(test_dataset[0]['cluster_uuids'])
print(f'n_neurons: {n_neurons}')
input_size = n_neurons * 100  # seq_len = 100

# DeTime related args
whole_dataset = concatenate_datasets([train_dataset, val_dataset, test_dataset])
max_time = max(whole_dataset['start_times'])
output_size = int(max_time-1) // args.t_res + 1


train_dataloader = make_loader(
    train_dataset,
    target='start_times',
    load_meta=True,
    batch_size=16,
    pad_to_right=True,
    pad_value=-1,
    bin_size=0.02,
    max_time_length=100,
    max_space_length=n_neurons,
    dataset_name='ibl',
    shuffle=True,
    start_time_up=max_time,
    dbin_size=args.t_res,
)

val_dataloader = make_loader(
    val_dataset,
    target='start_times',
    load_meta=True,
    batch_size=16,
    pad_to_right=True,
    pad_value=-1,
    bin_size=0.02,
    max_time_length=100,
    max_space_length=n_neurons,
    dataset_name='ibl',
    shuffle=False,
    start_time_up=max_time,
    dbin_size=args.t_res,
)

test_dataloader = make_loader(
    test_dataset,
    target='start_times',
    load_meta=True,
    batch_size=16,
    pad_to_right=True,
    pad_value=-1,
    bin_size=0.02,
    max_time_length=100,
    max_space_length=n_neurons,
    dataset_name='ibl',
    shuffle=False,
    start_time_up=max_time,
    dbin_size=args.t_res,
)

accelerator = Accelerator()

model = DeTime(
    model_name=args.model,
    input_size=input_size,
    output_size=output_size,
    hidden_size=args.hidden_size,
    dropout_rate=0.2,
)
model = accelerator.prepare(model)
print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=args.eps)

best_eval_loss = np.inf
best_eval_epoch = 0
for epoch in range(args.epochs):
    # train
    model.train()
    train_loss = 0
    train_examples = 0
    for batch in train_dataloader:
        batch = move_batch_to_device(batch, accelerator.device)
        spikes_flat = batch['spikes_data'].reshape(batch['spikes_data'].shape[0], -1)
        outputs = model(
            spikes=spikes_flat,
            target=batch['target']
        )
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        train_examples += outputs['n_examples']

    train_loss /= train_examples
    print(f"Epoch {epoch} training loss: {train_loss}")

    
    model.eval()
    eval_loss = 0
    eval_examples = 0
    gt, preds = [], []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = move_batch_to_device(batch, accelerator.device)
            spikes_flat = batch['spikes_data'].reshape(batch['spikes_data'].shape[0], -1)
            outputs = model(
                spikes=spikes_flat,
                target=batch['target']
            )
            loss = outputs['loss']
            eval_loss += loss.item()
            eval_examples += outputs['n_examples']
            preds.append(outputs['logits'])
            gt.append(batch['target'])

    eval_loss /= eval_examples
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        best_eval_epoch = epoch
        torch.save(model, os.path.join(save_path, 'model_best.pth'))
        
    print(f"Epoch {epoch} eval loss: {eval_loss}")

    gt = torch.cat(gt, dim=0)
    preds = torch.cat(preds, dim=0)

    gt_idx = (gt>0.5).sum(1).to(torch.float)
    preds_idx = (preds>0.5).sum(1).to(torch.float)
    print(f'gt: {gt_idx}\n preds: {preds_idx}')
    mse = metrics_list(
        gt=gt_idx, 
        pred=preds_idx,
        metrics=['mse'],
        device=accelerator.device
    )['mse']

    wandb.log({
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "mse": mse,
        "best_epoch": best_eval_epoch,
    })

torch.save(model, os.path.join(save_path, 'model_last.pth'))

wandb.log({
    "best_eval_loss": best_eval_loss
})

wandb.finish()
        
        
        
        








