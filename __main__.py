from math import floor
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

WINDOW_WIDTH = 700
WINDOW_HEIGT = 400
GRAPHIC_WIDTH = 650
GRAPHIC_HEIGHT = 20

structure_key_list = list(structure_dict.keys())
structure_val_list = list(structure_dict.values())

structure_color: dict = {
    'C': (0.5, 0.5, 0.5, 1.0),
    'H': (1.0, 0.0, 0.0, 1.0),
    'B': (0.0, 0.0, 1.0, 1.0)
}

dpg.create_context()
dpg.create_viewport(width=WINDOW_WIDTH, height=WINDOW_HEIGT, title='Protein secondary structure prediction tool')
dpg.setup_dearpygui()

device = "cpu"
model: NeuralNetwork = NeuralNetwork(M).to(device)
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device(device)))
model.eval()

def verify_input_sequence(input_sequence: str) -> bool:
    '''
    Verify that the given input sequence consists of valid amino acid codes.

    Parameters:
        input_sequence (str): A string of amino acid codes

    Returns:
        bool: True or False
    '''
    for char in input_sequence:
        if char not in residue_dict:
            return False
    
    return True

def delete_graphic_output() -> None:
    '''
    Delete all items in the graphic output window.
    '''
    dpg.delete_item("structure_texture")
    dpg.delete_item("confidence_texture")
    dpg.delete_item('graphic1')
    dpg.delete_item('graphic2')
    dpg.delete_item('graphic3')
    dpg.delete_item('graphic4')

def analyze_callback() -> None:
    '''
    Take the user input, run it through the prediction model, and display the results.
    '''
    sequence_input: str = dpg.get_value('sequence_input_text')
    sequence_input = sequence_input.upper()

    if not (verify_input_sequence(sequence_input)):
        dpg.set_value('sequence_input_error', 'Invalid input sequence')
        return
    else:
        dpg.set_value('sequence_input_error', '')
    
    sequence_tensor: Tensor = transform_sequence(sequence_input, SPAN)

    with torch.no_grad():
        pred: Tensor = model.classify(sequence_tensor.view(-1, 20 * model.m))

        structure_output: str = ''
        for index in pred.argmax(1):
            structure_output += structure_key_list[structure_val_list.index(index)]
        
        dpg.set_value('sequence_output_text', sequence_input)
        dpg.set_value('structure_output_text', structure_output)

        delete_graphic_output()

        structure_texture: list = []
        confidence_texture: list = []

        for i in range(GRAPHIC_HEIGHT):
            for j in range(GRAPHIC_WIDTH):
                x: int = floor((j / GRAPHIC_WIDTH) * len(sequence_input))
                c: tuple = structure_color[structure_output[x]]

                structure_texture.append(c[0])
                structure_texture.append(c[1])
                structure_texture.append(c[2])
                structure_texture.append(1.0)

                y: float = pred[x][pred[x].argmax(0)].item()

                confidence_texture.append(0.0)
                confidence_texture.append(y)
                confidence_texture.append(0.0)
                confidence_texture.append(1.0)

        with dpg.texture_registry():
            dpg.add_static_texture(width=GRAPHIC_WIDTH, height=GRAPHIC_HEIGHT, default_value=structure_texture, tag="structure_texture")
        
        with dpg.texture_registry():
            dpg.add_static_texture(width=GRAPHIC_WIDTH, height=GRAPHIC_HEIGHT, default_value=confidence_texture, tag="confidence_texture")

        dpg.add_text('Structure', tag='graphic1', parent='graphic_output_window')
        dpg.add_image("structure_texture", tag='graphic2', parent='graphic_output_window')
        dpg.add_text('Confidence', tag='graphic3', parent='graphic_output_window')
        dpg.add_image("confidence_texture", tag='graphic4', parent='graphic_output_window')


with dpg.window() as primary_window:
    with dpg.window(label='Input', pos=(0, 0), width=WINDOW_WIDTH, height=130):
        dpg.add_text('Sequence')
        dpg.add_input_text(tag='sequence_input_text')
        dpg.add_button(label='Analyze sequence', callback=analyze_callback)
        dpg.add_text('', tag='sequence_input_error', color=(255, 32, 32, 255))

    with dpg.window(label='Text output', pos=(0, 130), width=WINDOW_WIDTH, height=100, horizontal_scrollbar=True):
        dpg.add_text('', tag='sequence_output_text')
        dpg.add_text('', tag='structure_output_text')

    with dpg.window(label='Graphic output', tag='graphic_output_window', pos=(0, 230), width=WINDOW_WIDTH, height=170):
        pass

dpg.set_primary_window(primary_window, True)
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()