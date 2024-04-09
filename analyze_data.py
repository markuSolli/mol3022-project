'''Module for functions related to analyzis of data'''
import os
from dotenv import load_dotenv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from protein_structure_dataset import ProteinStructureDataset

load_dotenv()
SPAN: int = int(os.getenv('SPAN'))
BATCH_SIZE: int = int(os.getenv('BATCH_SIZE'))
M: int = SPAN * 2 + 1

def analyze_dataset(csv: str, figname: str) -> None:
    '''
    Validates the model against the given csv file and analyzes the result.
    A confusion matrix is plotted and saved, as well as some accuracy measures.

    Parameters:
        csv (str): The filepath of the csv file to analyze.
        figname (str): The filepath of the plot to store.
    '''
    data = ProteinStructureDataset(csv, SPAN, "cpu")
    dataloader: DataLoader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    strucure_count = [0, 0, 0]
    size = len(dataloader.dataset)

    for _, structures in dataloader:
        for structure in structures:
            strucure_count[structure.argmax(0)] += 1

    coil_weight = size / (strucure_count[0] * 3)
    helix_weight = size / (strucure_count[1] * 3)
    beta_weight = size / (strucure_count[2] * 3)

    print("Coil: " + str(coil_weight))
    print("Helix: " + str(helix_weight))
    print("Beta: " + str(beta_weight))

    _fig, ax = plt.subplots()

    classes = ['coil', 'helix', 'beta strand']
    bar_colors = ['tab:gray', 'tab:blue', 'tab:red']

    ax.bar(classes, strucure_count, color=bar_colors)

    ax.set_ylabel('count')
    ax.set_title('Class occurence count')

    plt.savefig(figname)
    plt.show()

if __name__== "__main__":
    analyze_dataset('data/training.csv', 'plots/training_class_count.png')
