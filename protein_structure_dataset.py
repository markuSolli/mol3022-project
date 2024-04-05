import pandas as pd
from torch.utils.data import Dataset
from utils import transform_sequences, transform_structures

class ProteinStructureDataset(Dataset):
    def __init__(self, data_src: str, span: int, device):
        dataframe = pd.read_csv(data_src)
        self.data = transform_sequences(dataframe['sequence'].values, span).to(device)
        self.label = transform_structures(dataframe['structure'].values).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
