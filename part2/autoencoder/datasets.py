from torch.utils.data import Dataset
import torch


class UnlabeledDataset(Dataset):
    """Dataset wrapping unlabeled data tensors.

    Each sample will be retrieved by indexing tensors along the first
    dimension.

    Arguments:
        data (Tensor): contains sample data.
    """
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __getitem__(self, index):
        return self.data[index, ...]
    
    def __len__(self):
        return self.data.shape[0]
