import sys
import os
from dotenv import load_dotenv

import torch
from torch import nn
from torch.nn.modules import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn import metrics

from neural_network import NeuralNetwork
from protein_structure_dataset import ProteinStructureDataset

load_dotenv()
SPAN: int = int(os.getenv('SPAN'))
BATCH_SIZE: int = int(os.getenv('BATCH_SIZE'))
M: int = SPAN * 2 + 1

LEARNING_RATE: float = 1e-1
EPOCHS: int = 3

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

WEIGHT = torch.tensor([0.522195, 1.477305, 2.450396])
loss_fn: CrossEntropyLoss = nn.CrossEntropyLoss(weight=WEIGHT)

def train_loop(dataloader: DataLoader, model: NeuralNetwork, loss_fn: Module, optimizer: Optimizer, loss_list: list[float], loss_x: list[int], iteration: int):
    size = len(dataloader)
    
    model.train()

    for batch, (sequence, structure) in enumerate(dataloader):
        pred = model(sequence.view(-1, 20 * model.m))

        loss = loss_fn(pred, structure)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 40 == 0:
            loss_list.append(loss.item())
            loss_x.append(iteration)
            iteration += 1

        if batch % 5000 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch:>5d}/{size:>5d}]")
    
    return iteration

def test_loop(dataloader: DataLoader, model: NeuralNetwork, score_list: list[float], score_x: list[int], iteration: int):
    model.eval()
    actual = []
    predicted = []

    with torch.no_grad():
        for sequence, structure in test_dataloader:
            pred = model.classify(sequence.view(-1, 20 * model.m))

            actual.extend(structure.argmax(1).tolist())
            predicted.extend(pred.argmax(1).tolist())
    
    actual = np.array(actual)
    predicted = np.array(predicted)

    F1_score = metrics.f1_score(actual, predicted, average='macro')

    print('F1 score: %.4f\n' % F1_score)

    score_list.append(F1_score)
    score_x.append(iteration)


def test_model():
    model.eval()
    actual = []
    predicted = []

    with torch.no_grad():
        for sequence, structure in test_dataloader:
            pred = model.classify(sequence.view(-1, 20 * model.m))

            actual.extend(structure.argmax(1).tolist())
            predicted.extend(pred.argmax(1).tolist())
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    confusion_matrix = metrics.confusion_matrix(actual, predicted) 

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Coil', 'Helix', 'Beta'])

    fig, ax = plt.subplots(figsize=(8, 4))

    cm_display.plot(ax=ax)

    accuracy = metrics.accuracy_score(actual, predicted)
    precision = metrics.precision_score(actual, predicted, average='macro')
    recall = metrics.recall_score(actual, predicted, average='macro')
    F1_score = metrics.f1_score(actual, predicted, average='macro')

    plot_text = 'Accuracy: %.4f\nPrecision: %.4f\nRecall: %.4f\nF1 score: %.4f'%(accuracy, precision, recall, F1_score)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.15)
    ax.text(1.45, 0.95, plot_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    plt.tight_layout()

    now = datetime.now()
    filename = "plots/confusion_matrix_" + now.strftime('%Y%m%d_%H%M') + ".png"
    plt.savefig(filename)
    plt.show()

def train_model():
    training_data = ProteinStructureDataset('data/training.csv', SPAN, device)
    train_dataloader: DataLoader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

    optimizer: SGD = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    loss_list: list[float] = []
    loss_x: list[int] = []
    score_list: list[float] = []
    score_x: list[int] = []
    i: int = 0

    test_loop(test_dataloader, model, score_list, score_x, 0)

    for t in range(EPOCHS):
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {t+1}: lr {lr:.6f}\n-------------------------------")
        i = train_loop(train_dataloader, model, loss_fn, optimizer, loss_list, loss_x, i)
        test_loop(test_dataloader, model, score_list, score_x, i - 1)

        scheduler.step()

    print("Training done!")

    fig, ax1 = plt.subplots(figsize=(9, 4))

    color = 'tab:blue'
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss', color=color)
    ax1.set_ylim([0, 1.6])
    ax1.plot(loss_x, loss_list, '.', color=color, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('F1 score', color=color)
    ax2.set_ylim([0.0, 1.0])
    ax2.plot(score_x, score_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plot_text = 'F1 score: %.4f\nLR: %.1e - %.1e\nSpan: %d\nOptimizer: SGD\nLoss: CrossEntropyLoss\nModel:\nLinear(20m, 10m)\nReLU()\nLinear(10m, m)\nReLU()\nLinear(m, 3)'%(score_list[-1], LEARNING_RATE, lr, SPAN)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.15)
    ax2.text(1.15, 0.95, plot_text, transform=ax2.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    plt.tight_layout()

    now = datetime.now()
    filename = "plots/plot_" + now.strftime('%Y%m%d_%H%M') + ".png"
    plt.savefig(filename)
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