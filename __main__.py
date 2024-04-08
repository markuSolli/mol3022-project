import os
from dotenv import load_dotenv

import torch
from torch import Tensor

import dearpygui.dearpygui as dpg
import numpy as np

from neural_network import NeuralNetwork
from utils import transform_sequence, residue_dict, structure_dict

load_dotenv()
SPAN: int = int(os.getenv('SPAN'))
M: int = SPAN * 2 + 1

structure_key_list = list(structure_dict.keys())
structure_val_list = list(structure_dict.values())

dpg.create_context()
dpg.create_viewport(width=700, height=400, title='Protein secondary structure prediction tool')
dpg.setup_dearpygui()

device = "cpu"
model: NeuralNetwork = NeuralNetwork(M).to(device)
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device(device)))
model.eval()

def verify_input_sequence(input_sequence: str) -> bool:
    for char in input_sequence.upper():
        if char not in residue_dict:
            return False
    
    return True

def analyze_callback(_, app_data, user_data):
    sequence_input: str = dpg.get_value('sequence_input_text')

    if not (verify_input_sequence(sequence_input)):
        dpg.set_value('sequence_input_error', 'Invalid input sequence')
        return
    else:
        dpg.set_value('sequence_input_error', '')
    
    sequence_tensor = transform_sequence(sequence_input, SPAN)

    with torch.no_grad():
        pred = model.classify(sequence_tensor.view(-1, 20 * model.m))

        structure_output = ''
        for index in pred.argmax(1):
            structure_output += structure_key_list[structure_val_list.index(index)]
        
        dpg.set_value('sequence_output_text', sequence_input)
        dpg.set_value('structure_output_text', structure_output)


with dpg.window() as primary_window:
    with dpg.window(label='Input', pos=(0, 0), width=700, height=130):
        dpg.add_text('Sequence')
        dpg.add_input_text(tag='sequence_input_text')
        dpg.add_button(label='Analyze sequence', callback=analyze_callback)
        dpg.add_text('', tag='sequence_input_error', color=(255, 32, 32, 255))

    with dpg.window(label='Text output', pos=(0, 130), width=700, height=130, horizontal_scrollbar=True):
        dpg.add_text('', tag='sequence_output_text')
        dpg.add_text('', tag='structure_output_text')

    with dpg.window(label='Graphic output', pos=(0, 260), width=700, height=140):
        texture_data = []
        for i in range(0, 700 * 20):
            texture_data.append(255 / 255)
            texture_data.append(0)
            texture_data.append(255 / 255)
            texture_data.append(255 / 255)

        with dpg.texture_registry(show=True):
            dpg.add_static_texture(width=700, height=20, default_value=texture_data, tag="texture_tag")

        dpg.add_image("texture_tag")            


dpg.set_primary_window(primary_window, True)
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()