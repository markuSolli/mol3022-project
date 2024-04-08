from torch import nn
from torch.nn.modules import Module

class NeuralNetwork(Module):
    """
    A Neural Network for use in predicting protein secondary structures based on its protein sequence.
    ...

    Attributes
    ----------
    m : int
        The amount of amino acid codes to account for in the prediction.

    Methods
    -------
    classify(x: Tensor):
        Perform a forward pass using x as input, and apply softmax before returning.
        Meant for validating or using the model.
    """
    def __init__(self, m: int):
        super().__init__()
        self.m = m
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20 * self.m, 10 * self.m),
            nn.ReLU(),
            nn.Linear(10 * self.m, self.m),
            nn.ReLU(),
            nn.Linear(self.m, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def classify(self, x):
        return self.softmax(self.forward(x))