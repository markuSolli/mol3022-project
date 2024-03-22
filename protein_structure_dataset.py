import pandas as pd
from torch.utils.data import Dataset
from utils import transform_sequence, transform_structure

class ProteinStructureDataset(Dataset):
    def __init__(self, data_src: str, span: int):
        self.data = pd.read_csv(data_src)
        self.span = span

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]
        
        data = transform_sequence(data, self.span)
        label = transform_structure(label)

        return data, label

def custom_collate_fn(batch):
    return tuple(zip(*batch))