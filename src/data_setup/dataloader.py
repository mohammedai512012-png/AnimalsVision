import torch
from torch.utils.data import DataLoader

def get_dataloader(dataset: torch.utils.data.Dataset,
                   batch_size: int = 32,
                   shuffle: bool = True,
                   num_workers: int = 4):
    """
    
    """
    
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=int(num_workers))


