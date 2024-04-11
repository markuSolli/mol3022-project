'''Module for functions related to training and validation of the neural network'''
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

import torch
from torch import Tensor, nn
from torch.nn.modules import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
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

DEVICE: str = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f'Using {DEVICE} device')

model: NeuralNetwork = NeuralNetwork(M).to(DEVICE)

test_data = ProteinStructureDataset('data/test.csv', SPAN, DEVICE)
test_dataloader: DataLoader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

train_data = ProteinStructureDataset('data/training.csv', SPAN, DEVICE)
train_dataloader: DataLoader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

optimizer: SGD = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

WEIGHT: Tensor = torch.tensor([float(os.getenv('WEIGHT_C')),
                               float(os.getenv('WEIGHT_H')),
                               float(os.getenv('WEIGHT_B'))])
loss_fn: CrossEntropyLoss = nn.CrossEntropyLoss(weight=WEIGHT)

def train_loop(loss_list: list[float], loss_x: list[int], iteration: int) -> int:
    '''
    Validates the model using the test_dataloader.
    Prints the F1 score, and registers it to the given score_list.

    Parameters:
        score_list (list[float]): The list of F1 scores to plot
        loss_x: (list[int]): The list of x-values paired to F1 scores.
        iteration (int): The current iteration value. Will be incremented and returned.
    
    Returns:
        int: the incremented iteration value
    '''
    size = len(train_dataloader)

    model.train()

    for batch, (sequence, structure) in enumerate(train_dataloader):
        pred = model.forward(sequence.view(-1, 20 * model.m))

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

def test_loop(score_list: list[float]) -> None:
    '''
    Validates the model using the test_dataloader.
    Prints the F1 score, and registers it to the given score_list.

    Parameters:
        score_list (list[float]): The list of F1 scores to plot
    '''
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

    f1_score = metrics.f1_score(actual, predicted, average='macro')

    print(f'F1 score: {f1_score:.4f}\n')

    score_list.append(f1_score)

def evaluate_results(actual: list, predicted: list) -> None:
    '''
    Use sklearn to evaluate the model with different accuracy measures.

    Parameters:
        actual (list): List of actual classes (0, 1, or 2)
        predicted (list): List of predicted classes, in same order as the actual list.
    '''
    accuracy: float = metrics.accuracy_score(actual, predicted)
    precision: np.ndarray = metrics.precision_score(actual, predicted, average=None)
    recall: np.ndarray = metrics.recall_score(actual, predicted, average=None)
    f1_score: np.ndarray = metrics.f1_score(actual, predicted, average=None)

    print(f'Accuracy: {accuracy:.4f}\n'
          f'Precision: {precision[0]:.4f}, {precision[1]:.4f}, {precision[2]:.4f}\n'
          f'Recall: {recall[0]:.4f}, {recall[1]:.4f}, {recall[2]:.4f}\n'
          f'F1 score: {f1_score[0]:.4f}, {f1_score[1]:.4f}, {f1_score[2]:.4f}')

def test_model() -> None:
    '''
    Use data from 'data/test.csv' to evaluate the model.
    Plots a confusion matrix when completed.
    '''
    model.eval()
    actual = []
    predicted = []

    with torch.no_grad():
        for sequence, structure in test_dataloader:
            pred: Tensor = model.classify(sequence.view(-1, 20 * model.m))

            actual.extend(structure.argmax(1).tolist())
            predicted.extend(pred.argmax(1).tolist())

    actual = np.array(actual)
    predicted = np.array(predicted)
    evaluate_results(actual, predicted)

def plot_training_loss(lr: float,
                       loss_list: list,
                       loss_x: list,
                       score_list: list,
                       score_x: list) -> None:
    '''
    Plot training loss along the same axis as the F1 score for each epoch.

    Parameters:
        lr (float): The final learning rate after training.
        loss_list (list): List of loss values to plot.
        loss_x (list): List of x values corresponding to the loss values.
        score_list (list): List of F1 scores to plot.
        score_x (list): List of x values corresponding to the F1 scores.
    '''
    _fig, ax1 = plt.subplots(figsize=(9, 4))

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

    plot_text = (f'F1 score: {score_list[-1]:.4f}\n'
                 f'LR: {LEARNING_RATE:.1e} - {lr:.1e}\n'
                 f'Span: {SPAN:d}\n'
                 'Optimizer: SGD\n'
                 'Loss: CrossEntropyLoss\n'
                 'Model:\nLinear(20m, 10m)\nReLU()\nLinear(10m, m)\nReLU()\nLinear(m, 3)')
    props = {'boxstyle':'round', 'facecolor':'wheat', 'alpha':0.15}
    ax2.text(1.15, 0.95, plot_text,
             transform=ax2.transAxes,
             fontsize=11,
             verticalalignment='top',
             bbox=props)
    plt.tight_layout()

    now = datetime.now()
    filename = "plots/plot_" + now.strftime('%Y%m%d_%H%M') + ".png"
    plt.savefig(filename)
    plt.show()

def train_model() -> None:
    '''
    Use training data to train the neural network.
    Plots loss and F1 score when completed.
    '''
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    loss_list: list[float] = []
    loss_x: list[int] = []
    score_list: list[float] = []
    score_x: list[int] = []
    i: int = 0

    test_loop(score_list)
    score_x.append(0)

    for t in range(EPOCHS):
        lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch {t+1}: lr {lr:.6f}\n-------------------------------')
        i = train_loop(loss_list, loss_x, i)
        test_loop(score_list)
        score_x.append(i)

        scheduler.step()

    print("Training done!")

    plot_training_loss(lr, loss_list, loss_x, score_list, score_x)

def main() -> None:
    '''
    Execute code based on the given command line arguments.

    Arguments:
        -l: Load model from 'model_weights.pth' and test it using test_model(). 
            If this is argument is not given, the model will be trained using train_model().
        -s: Save the model to 'model_weights.pth'
    '''
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

    sys.exit(0)

if __name__== "__main__":
    main()
