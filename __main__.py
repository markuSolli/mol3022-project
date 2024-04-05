import torch
from torch import Tensor

import dearpygui.dearpygui as dpg
import numpy as np

from neural_network import NeuralNetwork
from utils import residues
from model import M

dpg.create_context()
dpg.create_viewport(width=700, height=300, title='Protein secondary structure prediction tool')
dpg.setup_dearpygui()

model: NeuralNetwork = NeuralNetwork(M)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

def sequence_str_to_tensor(input_sequence: str) -> Tensor:
    sequence_matrix = np.zeros((len(input_sequence), 20), dtype=np.float32)

    for i in range(len(input_sequence)):
        sequence_matrix[i][residues[input_sequence[i]]] = 1.0
    
    return torch.from_numpy(sequence_matrix)

def verify_input_sequence(input_sequence: str) -> bool:
    for char in input_sequence.upper():
        if char not in residues:
            return False
    
    return True

def analyze_callback(_, app_data, user_data):
    sequence_input: str = dpg.get_value('sequence_input_text')

    if not (verify_input_sequence(sequence_input)):
        dpg.set_value('sequence_input_error', 'Invalid input sequence')
        return
    else:
        dpg.set_value('sequence_input_error', '')
    
    sequence_tensor = sequence_str_to_tensor(sequence_input)

with dpg.window() as primary_window:
    dpg.add_text('Sequence')
    dpg.add_input_text(tag='sequence_input_text')
    dpg.add_button(label='Analyze sequence', callback=analyze_callback)
    dpg.add_text('', tag='sequence_input_error', color=(255, 32, 32, 255))

dpg.set_primary_window(primary_window, True)
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()