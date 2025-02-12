import torch
from loader.base import BaseDataset, FlexibleDataset
from torch.utils.data import Sampler
import random
from collections import defaultdict


def make_loader(dataset,
                batch_size,
                target=None,
                pad_to_right=True,
                sort_by_depth=False,
                sort_by_region=False,
                pad_value=0.,
                max_time_length=5000,
                max_space_length=100,
                bin_size=0.02,
                brain_region='all',
                load_meta=False,
                dataset_name="ibl",
                shuffle=True,
                **kwargs,
               ):
    dataset = BaseDataset(dataset=dataset,
                          target=target,
                          pad_value=pad_value,
                          max_time_length=max_time_length,
                          max_space_length=max_space_length,
                          bin_size=bin_size,
                          pad_to_right=pad_to_right,
                          dataset_name=dataset_name,
                          sort_by_depth=sort_by_depth,
                          sort_by_region=sort_by_region,
                          brain_region=brain_region,
                          load_meta=load_meta,
                          **kwargs,
                          )
    print(f"len(dataset): {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)
    return dataloader


def make_loader_flex(dataset,
                batch_size,
                target=None,
                pad_to_right=True,
                sort_by_depth=False,
                sort_by_region=False,
                pad_value=0.,
                max_time_length=5000,
                bin_size=0.02,
                brain_region='all',
                load_meta=False,
                dataset_name="ibl",
                shuffle=True,
                **kwargs,
               ):
    '''Make dataloader without padding in space dimension, using EIDBatchSampler. '''
    
    flex_dataset = FlexibleDataset(dataset=dataset,
                          target=target,
                          pad_value=pad_value,
                          max_time_length=max_time_length,
                          bin_size=bin_size,
                          pad_to_right=pad_to_right,
                          dataset_name=dataset_name,
                          sort_by_depth=sort_by_depth,
                          sort_by_region=sort_by_region,
                          brain_region=brain_region,
                          load_meta=load_meta,
                          **kwargs,
                          )
    
    print(f"len(dataset): {len(flex_dataset)}")

    eid_batch_sampler = EIDBatchSampler(flex_dataset, batch_size=batch_size, shuffle=shuffle) 
    
    dataloader = torch.utils.data.DataLoader(flex_dataset, batch_sampler=eid_batch_sampler)
    
    
    return dataloader
    
    

class EIDBatchSampler(Sampler):
    """
    EID Batch Sampler. Make sure each batch only contains trials from 1 session. Trials within each batch will have the same EID field.
    Args:
        data_source: dataset,
        batch_size: batch size for mini-batch interation,
        shuffle: randomly or sequentially select batches.
    """
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle  # Shuffle groups and data within groups
        self.grouped_indices = self._group_by_eid()  # Group data by 'eid'

    # Group the dataset by 'eid'
    def _group_by_eid(self):
        groups = defaultdict(list)
        for idx, data in enumerate(self.data_source):
            groups[data['eid']].append(idx)
        return groups

    # Iterator: dynamic selection based on shuffle
    def __iter__(self):
        # Reset the state at the start of each epoch
        eids = list(self.grouped_indices.keys())  # List of group IDs
        
        # Shuffle group order if shuffle=True
        if self.shuffle:
            random.shuffle(eids)

        # Create a fresh copy of the grouped data for dynamic sampling
        grouped_data = {eid: indices[:] for eid, indices in self.grouped_indices.items()}
        
        # Keep selecting batches until all groups are exhausted
        while grouped_data:
            # Randomly (or sequentially) select a group
            if self.shuffle:
                current_eid = random.choice(eids)
            else:
                current_eid = eids[0]

            group = grouped_data[current_eid]

            # Select batch_size data from the group
            if len(group) <= self.batch_size:
                # If group has fewer than or equal to batch_size items, yield them all
                batch = group
                del grouped_data[current_eid]  # Remove the group from further selection
                eids.remove(current_eid)  # Also remove from eid list
            else:
                # If more than batch_size items, randomly (or sequentially) pick batch_size items
                if self.shuffle:
                    batch = random.sample(group, self.batch_size)
                else:
                    batch = group[:self.batch_size]
                
                # Remove selected items from the group
                grouped_data[current_eid] = [i for i in group if i not in batch]

            yield batch

    # Return the total number of batches (ceil division)
    def __len__(self):
        total_data = sum(len(indices) for indices in self.grouped_indices.values())
        return (total_data + self.batch_size - 1) // self.batch_size  # Ceil division
