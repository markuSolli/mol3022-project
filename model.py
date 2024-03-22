import sys

import torch
from torch import nn
from torch.nn.modules import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from protein_structure_dataset import ProteinStructureDataset, custom_collate_fn

SPAN = 6
M = SPAN * 2 + 1

LEARNING_RATE: float = 0.1
BATCH_SIZE: int = 64
EPOCHS: int = 5

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model: NeuralNetwork = NeuralNetwork(M).to(device)

test_data = ProteinStructureDataset('data/test.csv', SPAN)
test_dataloader: DataLoader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

loss_fn: CrossEntropyLoss = nn.CrossEntropyLoss()

def train_loop(dataloader: DataLoader, model: NeuralNetwork, loss_fn: Module, optimizer: Optimizer, loss_list: list[float], loss_x: list[int], iteration: int):
    size = len(dataloader.dataset)
    
    model.train()

    for batch, (sequences, structures) in enumerate(dataloader):
        batch_loss = 0
        batch_counter = 0

        for i in range(len(sequences)):
            sequence_loss = 0.0

            for j in range(len(sequences[i]) - model.m + 1):
                pred = model(sequences[i][j:j+model.m].view(-1, 20 * model.m))

                loss = loss_fn(pred[0], structures[i][j])
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                batch_loss += loss.item()
                batch_counter += 1

                sequence_loss += loss.item()
            
            sequence_loss /= (len(sequences[i]) - model.m + 1)
            loss_list.append(sequence_loss)

            loss_x.append(iteration)
            iteration += 1

        loss = batch_loss / batch_counter
        current = batch * BATCH_SIZE + len(sequences)
        print(f"loss: {loss:.5f}  [{current:>5d}/{size:>5d}]")
    
    return iteration

def test_loop(dataloader: DataLoader, model: NeuralNetwork, loss_fn: Module, accuracy_list: list[float], accuracy_x: list[int], iteration: int):
    model.eval()
    counter: int = 0
    correct = 0

    with torch.no_grad():
        for sequences, structures in dataloader:
            for i in range(len(sequences)):
                for j in range(len(sequences[i]) - model.m + 1):
                    pred = model.classify(sequences[i][j:j+model.m].view(-1, 20 * model.m))

                    correct += (structures[i][j][pred.argmax(1)] == 1).type(torch.float).sum().item()

                    counter += 1

    correct /= counter
    print(f"Accuracy: {(100*correct):.1f}% \n")

    accuracy_list.append(100*correct)
    accuracy_x.append(iteration)

def test_model():
    test_loop(test_dataloader, model, loss_fn)

    print("Testing done!")

def train_model():
    training_data = ProteinStructureDataset('data/training.csv', SPAN)
    train_dataloader: DataLoader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

    optimizer: SGD = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=(EPOCHS - 1))

    loss_list: list[float] = []
    loss_x: list[int] = []
    accuracy_list: list[float] = []
    accuracy_x: list[int] = []
    i: int = 0

    for t in range(EPOCHS):
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {t+1}: lr {lr:.4f}\n-------------------------------")
        i = train_loop(train_dataloader, model, loss_fn, optimizer, loss_list, loss_x, i)
        test_loop(test_dataloader, model, loss_fn, accuracy_list, accuracy_x, i - 1)

        scheduler.step()

    print("Training done!")

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(loss_x, loss_list, '.', color=color, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('accuracy (%)', color=color)
    ax2.set_ylim([0, 100])
    ax2.plot(accuracy_x, accuracy_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

def main():
    arguments: list[str] = sys.argv[1:]
    load: bool = False
    save: bool = False

    for argument in arguments:
        match argument:
            case '-l': load = True
            case '-s': save = True
    
    if load:
        model.load_state_dict(torch.load('model_weights.pth'))
        test_model()
    else:
        train_model()
    
    if save:
        torch.save(model.state_dict(), 'model_weights.pth')

    exit(0)

if __name__== "__main__":
    main()