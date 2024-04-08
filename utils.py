import torch
from torch import Tensor
import numpy as np

residue_dict: dict[str, int] = {
    'C': 0,
    'S': 1,
    'T': 2,
    'P': 3,
    'A': 4,
    'G': 5,
    'N': 6,
    'D': 7,
    'E': 8,
    'Q': 9,
    'H': 10,
    'R': 11,
    'K': 12,
    'M': 13,
    'I': 14,
    'L': 15,
    'V': 16,
    'F': 17,
    'Y': 18,
    'W': 19
}

structure_dict: dict[str, int] = {
    'C': 0,
    'H': 1,
    'B': 2
}

def transform_sequence(sequence: str, span: int) -> Tensor:
    '''
    Takes a single sequence string and transforms it to a Tensor, ready for use in the model.
    Since protein sequences are of varying length, it chops the string into pieces of (2 * span + 1) length.
    To make endpoints valid, it applies zero padding to the edges.

    Parameters:
        sequences: The protein sequence strings to transform.
        span (int): The padding to apply left and right of the sequence.
    
    Returns:
        Tensor: The constructed Tensor consisting of the transformed structure matrices.
    '''
    t_sequences = []
    m = span * 2 + 1

    sequence_matrix = np.zeros((len(sequence) + span * 2, 20), dtype=np.float32)

    for i in range(len(sequence)):
        sequence_matrix[i + span][residue_dict[sequence[i]]] = 1.0
        
    for i in range(len(sequence)):
        t_sequences.append(sequence_matrix[i:i+m])
    
    return torch.from_numpy(np.array(t_sequences))

def transform_sequences(sequences, span: int) -> Tensor:
    '''
    Takes a list of sequence string and transforms it to a Tensor, ready for use in the model.
    Since protein sequences are of varying length, it chops the string into pieces of (2 * span + 1) length.
    To make endpoints valid, it applies zero padding to the edges.

    Parameters:
        sequences: The list of protein sequence strings to transform.
        span (int): The padding to apply left and right of the sequence.
    
    Returns:
        Tensor: The constructed Tensor consisting of the transformed structure matrices.
    '''
    t_sequences = []
    m = span * 2 + 1

    for sequence in sequences:
        sequence_matrix = np.zeros((len(sequence) + span * 2, 20), dtype=np.float32)

        for i in range(len(sequence)):
            sequence_matrix[i + span][residue_dict[sequence[i]]] = 1.0
        
        for i in range(len(sequence)):
            t_sequences.append(sequence_matrix[i:i+m])
        
    return torch.from_numpy(np.array(t_sequences))

def transform_structures(structures) -> Tensor:
    '''
    Takes a list of structure string and transforms it to a Tensor, ready for use in the model.
    A structure matrix has a size of 3, and one of the elements is 1 corresponding to the matching structure.

    Parameters:
        structures: The list of structure string to transform.
    
    Returns:
        Tensor: The constructed Tensor consisting of a BATCH_SIZE amount of structure matrices.
    '''
    t_structures = []

    for structure in structures:
        for i in range(len(structure)):
            structure_matrix = np.zeros(3, dtype=np.float32)
            structure_matrix[structure_dict[structure[i]]] = 1.0
        
            t_structures.append(structure_matrix)
    
    return torch.from_numpy(np.array(t_structures))
