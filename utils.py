import torch
import numpy as np

residues: dict[str, int] = {
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

structures: dict[str, int] = {
    'C': 0,
    'H': 1,
    'B': 2
}

def transform_sequence(sequence: str, span: int):
    sequence_matrix = np.zeros((len(sequence) + span * 2, 20), dtype=np.float32)

    for i in range(span, len(sequence) - span):
        sequence_matrix[i][residues[sequence[i]]] = 1.0
    
    return torch.from_numpy(sequence_matrix)

def transform_structure(structure: str):
    structure_matrix = np.zeros((len(structure), 3), dtype=np.float32)

    for i in range(len(structure)):
        structure_matrix[i][structures[structure[i]]] = 1.0
    
    return torch.from_numpy(structure_matrix)
