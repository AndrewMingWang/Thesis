import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os
from os.path import isfile, join
from os import listdir

class ProcessedMeshDataset(Dataset):
    """
    PyTorch dataset containing processed mesh data.
    Each data point is an (# faces by 7 by 11) local patch feature tensor
    """

    def __init__(self, processed_data_folder, labels_file):
        # Read file containing labels and separate each line
        f = open(labels_file, "r")
        labels_lines = f.read().splitlines()

        # Create label_dict to map a mesh's idx to its label
        # labels_dict is {00024: 7, 00015: 1, ...} ie: mesh_idx: label
        self.labels_dict = {}
        for line in labels_lines:
            self.labels_dict[int(line[:-2])] = int(line[-1])

        # Count how many occurrences of each class we have
        self.class_counts = {}
        for line in labels_lines:
            label = int(line[-1])
            if label in self.class_counts:
                self.class_counts[label] += 1
            else:
                self.class_counts[label] = 1

        # Get filenames for processed tet meshes
        saved_patch_tensor_files = [join(processed_data_folder, f) for f in listdir(processed_data_folder)
                                    if (isfile(join(processed_data_folder, f)) and f[-3:] == ".pt")]

        # Get total number of data points
        self.num_tensors = len(saved_patch_tensor_files)

        # Load meshes into memory
        self.mesh_tensors = [torch.load(f) for f in saved_patch_tensor_files]

        # Get mesh idx for each mesh (for mapping mesh to label)
        self.mesh_indices = [int(f[-8:-3]) for f in saved_patch_tensor_files]

    def __len__(self):
        return self.num_tensors

    def __getitem__(self, idx):
        mesh = self.mesh_tensors[idx]
        mesh_idx = self.mesh_indices[idx]

        label = self.labels_dict[mesh_idx]

        return mesh, label
