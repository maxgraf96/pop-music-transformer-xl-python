# =============================================
# Simple custom dataset: Take data generated in "data_loading.py" and wrap it in a pytorch tensor
# =============================================

import torch
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    """
    Custom data set to be used with the segments generated in "data_loading.py"
    """
    def __init__(self, segments):
        """
        Segments coming from "data_loading.py"
        :param segments:
        """
        self.segments = segments

    def __len__(self):
        """
        :return: Number of segments
        """
        return len(self.segments)

    def __getitem__(self, idx):
        """
        Convert item to tensor
        :param idx: Item index
        :return: Dict with tensor
        """
        segments = self.segments[idx]
        sample = {"segments": torch.tensor(segments).to(device)}
        return sample