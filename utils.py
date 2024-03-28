import torch
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

def transform_sequence(sequences, span: int):
    t_sequences = []
    m = span * 2 + 1

    for sequence in sequences:
        sequence_matrix = np.zeros((len(sequence) + span * 2, 20), dtype=np.float32)

        for i in range(span, len(sequence) - span):
            sequence_matrix[i][residue_dict[sequence[i]]] = 1.0
        
        for i in range(len(sequence_matrix) - m + 1):
            t_sequences.append(sequence_matrix[i:i+m])
        
    return torch.from_numpy(np.array(t_sequences))

def transform_structure(structures):
    t_structures = []

    for structure in structures:
        for i in range(len(structure)):
            structure_matrix = np.zeros(3, dtype=np.float32)
            structure_matrix[structure_dict[structure[i]]] = 1.0
        
            t_structures.append(structure_matrix)
    
    return torch.from_numpy(np.array(t_structures))
