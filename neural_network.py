from torch import nn
from torch.nn.modules import Module

class NeuralNetwork(Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20 * self.m, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def classify(self, x):
        return self.softmax(self.forward(x))