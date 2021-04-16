import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os
from os.path import isfile, join
from os import listdir
import pickle

class Thingi10kBinaryVoxelDataset(Dataset):
    """
    PyTorch dataset containing voxelized mesh data
    Each data point is a (50 by 50 by 50) sized tensor
    """

    def __init__(self, voxelized_data_folder, labels_file):
        # Read labels dictionary
        with (open(labels_file, "rb")) as f:
            self.labels_dict = pickle.load(f)

        # Count class occurences
        self.class_counts = {0: 0, 1: 0}
        for key in self.labels_dict:
            self.class_counts[self.labels_dict[key]] += 1

        # Get filenames for processed tet meshes
        voxelized_tensor_files = [join(voxelized_data_folder, f) for f in listdir(voxelized_data_folder)
                                    if (isfile(join(voxelized_data_folder, f)) and f[-3:] == ".pt")]

        relevant_tensor_files = []
        # Only take relevant meshes
        for f in voxelized_tensor_files:
            mesh_idx = int(f[-7:-3])
            if mesh_idx in self.labels_dict.keys():
                relevant_tensor_files.append(f)

        # Get total number of data points
        self.num_tensors = len(relevant_tensor_files)

        # Load meshes into memory
        self.mesh_tensors = [torch.load(f) for f in relevant_tensor_files]

        # Get mesh idx for each mesh (for mapping mesh to label)
        self.mesh_indices = [int(f[-7:-3]) for f in relevant_tensor_files]

        # Count how many occurrences of each class we have
        self.class_counts = {0: 0, 1: 0}
        for f in relevant_tensor_files:
            mesh_idx = int(f[-7:-3])
            self.labels_dict[mesh_idx]
            self.class_counts[self.labels_dict[mesh_idx]] += 1

    def __len__(self):
        return self.num_tensors

    def __getitem__(self, idx):
        mesh = self.mesh_tensors[idx]
        mesh_idx = self.mesh_indices[idx]

        label = self.labels_dict[mesh_idx]

        return mesh, label

class VoxelDataset(Dataset):
    """
    PyTorch dataset containing voxelized mesh data
    Each data point is a (25 by 25 by 10) sized tensor
    """

    def __init__(self, voxelized_data_folder, labels_file):
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
        voxelized_tensor_files = [join(voxelized_data_folder, f) for f in listdir(voxelized_data_folder)
                                    if (isfile(join(voxelized_data_folder, f)) and f[-3:] == ".pt")]

        # Get total number of data points
        self.num_tensors = len(voxelized_tensor_files)

        # Load meshes into memory
        self.mesh_tensors = [torch.load(f) for f in voxelized_tensor_files]

        # Get mesh idx for each mesh (for mapping mesh to label)
        self.mesh_indices = [int(f[-8:-3]) for f in voxelized_tensor_files]

    def __len__(self):
        return self.num_tensors

    def __getitem__(self, idx):
        mesh = self.mesh_tensors[idx]
        mesh_idx = self.mesh_indices[idx]

        label = self.labels_dict[mesh_idx]

        return mesh, label

class BinaryVoxelDataset(Dataset):
    """
    PyTorch dataset containing voxelized mesh data
    Each data point is a (25 by 25 by 10) sized tensor
    """

    def __init__(self, voxelized_data_folder, labels_file, label0=0, label1=1):
        # Read file containing labels and separate each line
        f = open(labels_file, "r")
        labels_lines = f.read().splitlines()

        # Create label_dict to map a mesh's idx to its label
        # labels_dict is {00024: 7, 00015: 1, ...} ie: mesh_idx: label
        self.labels_dict = {}
        for line in labels_lines:
            mesh_idx = int(line[:-2])
            label = int(line[-1])

            # Only add to dict if label is label0 or label1
            if label == label0:
                self.labels_dict[mesh_idx] = 0
            elif label == label1:
                self.labels_dict[mesh_idx] = 1

        # Get filenames for processed tet meshes
        voxelized_tensor_files = [join(voxelized_data_folder, f) for f in listdir(voxelized_data_folder)
                                    if (isfile(join(voxelized_data_folder, f)) and f[-3:] == ".pt")]

        relevant_tensor_files = []
        # Only take relevant meshes
        for f in voxelized_tensor_files:
            mesh_idx = int(f[-8:-3])
            if mesh_idx in self.labels_dict.keys():
                relevant_tensor_files.append(f)

        # Get total number of data points
        self.num_tensors = len(relevant_tensor_files)

        # Load meshes into memory
        self.mesh_tensors = [torch.load(f) for f in relevant_tensor_files]

        # Get mesh idx for each mesh (for mapping mesh to label)
        self.mesh_indices = [int(f[-8:-3]) for f in relevant_tensor_files]

        # Count how many occurrences of each class we have
        self.class_counts = {0: 0, 1: 0}
        for f in relevant_tensor_files:
            mesh_idx = int(f[-8:-3])
            self.labels_dict[mesh_idx]
            self.class_counts[self.labels_dict[mesh_idx]] += 1

    def __len__(self):
        return self.num_tensors

    def __getitem__(self, idx):
        mesh = self.mesh_tensors[idx]
        mesh_idx = self.mesh_indices[idx]

        label = self.labels_dict[mesh_idx]

        return mesh, label