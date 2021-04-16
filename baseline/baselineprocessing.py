import os
from os.path import isfile, join
from os import listdir
from itertools import permutations

import torch
import igl
import re
import numpy as np
import math
import time

import matplotlib.pyplot as plt
from pylab import figure
from mpl_toolkits.mplot3d import Axes3D
from meshplot import plot, subplot, interact
import meshplot

def get_ranges(data_folder):
    xrange_max = 0
    yrange_max = 0
    zrange_max = 0
    meshes = [join(data_folder, f) for f in listdir(data_folder) if (isfile(join(data_folder, f)) and f[-4:] == ".msh")]

    for mesh in meshes:
        verts, tets = igl.read_msh(mesh)
        xmin = np.min(verts[:, 0])
        ymin = np.min(verts[:, 1])
        zmin = np.min(verts[:, 2])
        xmax = np.max(verts[:, 0])
        ymax = np.max(verts[:, 1])
        zmax = np.max(verts[:, 2])

        xrange = abs(xmax - xmin)
        yrange = abs(ymax - ymin)
        zrange = abs(zmax - zmin)

        if xrange > xrange_max:
            xrange_max = xrange
        if yrange > yrange_max:
            yrange_max = yrange
        if zrange > zrange_max:
            zrange_max = zrange

    print(xrange_max)
    print(yrange_max)
    print(zrange_max)


def zero_meshes(data_folder, output_folder):
    """
    finds the minimum x, y, z values in the positions of the vertices and subtracts
    these extremal values from all points. Then divides them by their maxes. Results
    in all vertices in a mesh being within a [0, 1] for all dimensions.
    Reads files from "data_folder" and outputs zeroed meshes to "output_folder".
    :param data_folder: reads meshes of format xxxx.msh from this path
    :param output_folder: writes mesh vertices as (num_vertices, 3) tensors xxxx.pt to this path
    :return:
    """

    meshes = [join(data_folder, f) for f in listdir(data_folder) if (isfile(join(data_folder, f)) and f[-4:] == ".msh")]
    for mesh in meshes:
        verts, _ = igl.read_msh(mesh)
        xmin = np.min(verts[:, 0])
        ymin = np.min(verts[:, 1])
        zmin = np.min(verts[:, 2])
        xmax = np.max(verts[:, 0])
        ymax = np.max(verts[:, 1])
        zmax = np.max(verts[:, 2])
        xrange = abs(xmax - xmin)
        yrange = abs(ymax - ymin)
        zrange = abs(zmax - zmin)

        # Make minimum in all dimensions zero
        zeroer = np.array([xmin, ymin, zmin])
        zeroed = verts - zeroer

        # Make range in all dimensions 1
        divisor = np.array([xrange, yrange, zrange])
        res = zeroed / divisor

        # Save as tensor
        to_save = torch.from_numpy(res)
        mesh_idx = mesh[-8:-4] # -9 for MNIST data
        torch.save(to_save, output_folder + "/" + mesh_idx + ".pt")

def create_voxelized_data(size,
                          zeroed_data_folder,
                          voxelized_data_folder):
    """
    Creates a grid of voxels of size "size". Increments each voxel if a point is within it.
    :param size: create a voxelized mesh of size (size.x, size.y, size.z)
    :param zeroed_data_folder: folder with the zeroed mesh data as tensors ".pt"
    :param voxelized_data_folder: output folder to save the voxelized data tensors ".pt"
    :return:
    """

    zeroed_data = [join(zeroed_data_folder, f) for f in listdir(zeroed_data_folder)
                   if (isfile(join(zeroed_data_folder, f)) and f[-3:] == ".pt")]

    multiplier = torch.tensor((size[0] - 1, size[1] - 1, size[2] - 1))
    for mesh in zeroed_data:
        data = torch.load(mesh)
        res = torch.zeros(size)

        # Multiply by dimensions in size
        data *= multiplier

        # Get indices of voxels
        indices = torch.floor(data).numpy().astype(int)

        # Contruct voxelized mesh
        for idx in indices:
            res[idx[0], idx[1], idx[2]] = 1

        mesh_idx = mesh[-7:-3]
        torch.save(res, voxelized_data_folder + "/" + mesh_idx + ".pt")

if __name__ == "__main__":
    data_folder = '../data/Thingi10k/raw'
    zeroed_data_folder = '../data/Thingi10k/baseline_data/zeroed'
    voxelized_data_folder = '../data/Thingi10k/baseline_data/voxelized'

    zero = False
    voxelize = True

    # get_ranges(data_folder)
    # MNIST
    # Ranges are 20.7, 20.7, 4.8 -> 2000 voxels
    # Thingi10k
    # Ranges are 497, 571, 485 -> 50x50x50 125000 voxels

    if zero:
        zero_meshes(data_folder, zeroed_data_folder)
    if voxelize:
        create_voxelized_data((50, 50, 50), zeroed_data_folder, voxelized_data_folder)

