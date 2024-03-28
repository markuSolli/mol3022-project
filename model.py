import sys

import torch
from torch import nn
from torch.nn.modules import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from datetime import datetime

from neural_network import NeuralNetwork
from protein_structure_dataset import ProteinStructureDataset

SPAN = 6
M = SPAN * 2 + 1

LEARNING_RATE: float = 0.1
BATCH_SIZE: int = 64
EPOCHS: int = 10

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model: NeuralNetwork = NeuralNetwork(M).to(device)

test_data = ProteinStructureDataset('data/test.csv', SPAN, device)
test_dataloader: DataLoader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

loss_fn: CrossEntropyLoss = nn.CrossEntropyLoss()

def train_loop(dataloader: DataLoader, model: NeuralNetwork, loss_fn: Module, optimizer: Optimizer, loss_list: list[float], loss_x: list[int], iteration: int):
    size = len(dataloader)
    
    model.train()

    for batch, (sequence, structure) in enumerate(dataloader):
        pred = model(sequence.view(-1, 20 * model.m))

        loss = loss_fn(pred, structure)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 50 == 0:
            loss_list.append(loss.item())
            loss_x.append(iteration)
            iteration += 1

        if batch % 5000 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch:>5d}/{size:>5d}]")
    
    return iteration

def test_loop(dataloader: DataLoader, model: NeuralNetwork, accuracy_list: list[float], accuracy_x: list[int], iteration: int):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0

    with torch.no_grad():
        for sequence, structure in dataloader:
            pred = model.classify(sequence.view(-1, 20 * model.m))

            correct += (structure[0][pred.argmax(1)] == 1).type(torch.float).sum().item()

    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}% \n")

    accuracy_list.append(100*correct)
    accuracy_x.append(iteration)

def test_model():
    test_loop(test_dataloader, model, [], [], 0)

    print("Testing done!")

def train_model():
    training_data = ProteinStructureDataset('data/training.csv', SPAN, device)
    train_dataloader: DataLoader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

    optimizer: SGD = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=(EPOCHS - 1))

    loss_list: list[float] = []
    loss_x: list[int] = []
    accuracy_list: list[float] = []
    accuracy_x: list[int] = []
    i: int = 0

    for t in range(EPOCHS):
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {t+1}: lr {lr:.4f}\n-------------------------------")
        i = train_loop(train_dataloader, model, loss_fn, optimizer, loss_list, loss_x, i)
        test_loop(test_dataloader, model, accuracy_list, accuracy_x, i - 1)

        scheduler.step()

    print("Training done!")

    fig, ax1 = plt.subplots(figsize=(9, 4))

    color = 'tab:blue'
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(loss_x, loss_list, '.', color=color, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('accuracy (%)', color=color)
    ax2.set_ylim([0, 100])
    ax2.plot(accuracy_x, accuracy_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plot_text = 'LR: %.4f-%.4f\nSpan: %d\nOptimizer: SGD\nLoss: CrossEntropyLoss\nModel:\nLinear(20, 20)\nLeakReLu\nLinear(20,10)\nLeakyReLu\nLinear(10,3)'%(LEARNING_RATE, lr, SPAN)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.15)
    ax2.text(1.15, 0.95, plot_text, transform=ax2.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    plt.tight_layout()

    now = datetime.now()
    filename = "plots/plot_" + now.strftime('%Y%m%d_%H%M') + ".png"
    plt.savefig(filename)
    print(filename)
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