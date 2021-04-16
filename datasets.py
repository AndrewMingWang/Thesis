import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

import os
from os.path import isfile, join
from os import listdir

class Thingi10kProcessedMeshDataset(Dataset):
    """
    PyTorch dataset containing processed mesh data.
    Each data point is a (#faces by 7 by 12) local patch feature tensor
    """

    def __init__(self, processed_data_folder, adjacency_data_folder, labels_file):
        # Read labels dictionary
        with (open(labels_file, "rb")) as f:
            self.labels_dict = pickle.load(f)

        # Count class occurences
        self.class_counts = {0:0, 1:0}
        for key in self.labels_dict:
            self.class_counts[self.labels_dict[key]] += 1

        # Get filenames for processed tet meshes
        saved_patch_tensor_files = [join(processed_data_folder, f) for f in listdir(processed_data_folder)
                                    if (isfile(join(processed_data_folder, f))
                                        and f[-3:] == ".pt")
                                    and f[-4] != "a"]

        # Get total number of data points
        self.num_tensors = len(saved_patch_tensor_files)

        # Load meshes into memory
        self.mesh_tensors = [torch.load(f).permute(2, 0, 1) for f in saved_patch_tensor_files]

        # Get mesh idx for each mesh (for mapping mesh to label)
        self.mesh_idx_list = [int(f[-7:-3]) for f in saved_patch_tensor_files]
        self.mesh_idx_set = set(self.mesh_idx_list)

        # Get adjacency data
        adjacency_files = [join(adjacency_data_folder, f) for f in listdir(adjacency_data_folder)
                           if isfile(join(adjacency_data_folder, f))
                               and f[-4:] == "a.pt"
                               and int(f[-8:-4]) in self.mesh_idx_set]

        # Make adjacency dictionary
        adjacency_keys = [int(f[-8:-4]) for f in adjacency_files]
        adjacency_values = [torch.unsqueeze(torch.load(f), 0).long() for f in adjacency_files]

        # Refactor -1 values in adjacencies to max_faces - 1
        # In this case this number is 14000 - 1 = 13999
        for adjacency in adjacency_values:
            adjacency[adjacency == -1] = adjacency.shape[1] - 1

        self.adjacency_dict = dict(zip(adjacency_keys, adjacency_values))

    def __len__(self):
        return self.num_tensors

    def __getitem__(self, idx):
        mesh = self.mesh_tensors[idx]
        mesh_idx = self.mesh_idx_list[idx]

        adjacency = self.adjacency_dict[mesh_idx]

        label = self.labels_dict[mesh_idx]

        return (mesh, adjacency), (label, idx)


class MNISTProcessedMeshDataset(Dataset):
    """
    PyTorch dataset containing processed mesh data.
    Each data point is an (# faces by 7 by 12) local patch feature tensor
    """

    def __init__(self, processed_data_folder, adjacency_data_folder, labels_file):
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
                                    if (isfile(join(processed_data_folder, f))
                                    and f[-3:] == ".pt")
                                    and f[-4] != "a"]

        # Get total number of data points
        self.num_tensors = len(saved_patch_tensor_files)

        # Load meshes into memory
        self.mesh_tensors = [torch.load(f).permute(2, 0, 1) for f in saved_patch_tensor_files]

        # Get mesh idx for each mesh (for mapping mesh to label)
        self.mesh_idx_list = [int(f[-8:-3]) for f in saved_patch_tensor_files]
        self.mesh_idx_set = set(self.mesh_idx_list)

        # Get adjacency data
        adjacency_files = [join(adjacency_data_folder, f) for f in listdir(adjacency_data_folder)
                           if isfile(join(adjacency_data_folder, f))
                               and f[-4:] == "a.pt"
                               and int(f[-9:-4]) in self.mesh_idx_set]

        # Make adjacency dictionary
        adjacency_keys = [int(f[-9:-4]) for f in adjacency_files]
        adjacency_values = [torch.unsqueeze(torch.load(f), 0).long() for f in adjacency_files]

        # Refactor -1 values in adjacencies to max_faces - 1
        # In this case this number is 9000 - 1 = 8999
        for adjacency in adjacency_values:
            adjacency[adjacency == -1] = adjacency.shape[1] - 1

        self.adjacency_dict = dict(zip(adjacency_keys, adjacency_values))




    def __len__(self):
        return self.num_tensors

    def __getitem__(self, idx):
        mesh = self.mesh_tensors[idx]
        mesh_idx = self.mesh_idx_list[idx]

        adjacency = self.adjacency_dict[mesh_idx]

        label = self.labels_dict[mesh_idx]

        return (mesh, adjacency), (label, idx)

class MNISTBinaryProcessedMeshDataset(Dataset):
    """
    PyTorch dataset containing processed mesh data.
    Each data point is an (# faces by 7 by 12) local patch feature tensor
    """

    def __init__(self, processed_data_folder, adjacency_data_folder, labels_file, label0=0, label1=1):
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
        saved_patch_tensor_files = [join(processed_data_folder, f) for f in listdir(processed_data_folder)
                                    if (isfile(join(processed_data_folder, f))
                                        and f[-3:] == ".pt")
                                        and f[-4] != "a"]

        relevant_patch_tensor_files = []
        # Only take relevant meshes
        for f in saved_patch_tensor_files:
            mesh_idx = int(f[-8:-3])
            if mesh_idx in self.labels_dict.keys():
                relevant_patch_tensor_files.append(f)

        # Load meshes into memory
        self.mesh_tensors = [torch.load(f).permute(2, 0, 1) for f in relevant_patch_tensor_files]

        # Get mesh idx for each mesh (for mapping mesh to label)
        self.mesh_idx_list = [int(f[-8:-3]) for f in relevant_patch_tensor_files]
        self.mesh_idx_set = set(self.mesh_idx_list)

        # Count how many occurrences of each class we have
        self.class_counts = {0:0, 1:0}
        for f in relevant_patch_tensor_files:
            mesh_idx = int(f[-8:-3])
            self.class_counts[self.labels_dict[mesh_idx]] += 1

        # Get total number of data points
        self.num_tensors = len(relevant_patch_tensor_files)

        # Get number of triangles per mesh
        self.num_triangles = self.mesh_tensors[0].shape[0]

        # Get adjacency data
        adjacency_files = [join(adjacency_data_folder, f) for f in listdir(adjacency_data_folder)
                           if isfile(join(adjacency_data_folder, f))
                               and f[-4:] == "a.pt"
                               and int(f[-9:-4]) in self.mesh_idx_set]

        # Make adjacency dictionary
        adjacency_keys = [int(f[-9:-4]) for f in adjacency_files]
        adjacency_values = [torch.unsqueeze(torch.load(f), 0) for f in adjacency_files]

        # Refactor -1 values in adjacencies to max_faces - 1
        # In this case this number is 9000 - 1 = 8999
        for adjacency in adjacency_values:
            adjacency[adjacency == -1] = adjacency.shape[1] - 1

        self.adjacency_dict = dict(zip(adjacency_keys, adjacency_values))

    def __len__(self):
        return self.num_tensors

    def __getitem__(self, idx):
        mesh = self.mesh_tensors[idx]
        mesh_idx = self.mesh_idx_list[idx]

        adjacency = self.adjacency_dict[mesh_idx]

        label = self.labels_dict[mesh_idx]

        return (mesh, adjacency), (label, idx)