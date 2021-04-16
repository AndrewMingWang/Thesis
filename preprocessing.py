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

def tri_to_str(tri):
    return "{}.{}.{}".format(tri[0], tri[1], tri[2])

def str_to_tri(tri_string):
    return [int(num) for num in re.findall('(\d+)\.(\d+)\.(\d+)', tri_string)[0]]

def process_mesh(mesh_file, output_folder):
    """
    We define a neighbourhood as a face and its two coincident tets. Each
    neighbourhood is comprised of 7 faces with 5 vertices. Faces on the surface
    of the tetmesh have only a single tet in this case 4 faces and 4 vertices.
    We uniquely order the vertices as v0, v1, v2, pole0 and pole1. The ordering
    of v0, v1, v2 is determined by angle (largest to smallest). Poles are ordered
    by distance to the triangle defined by v0, v1, v2.

    Takes each mesh and produces two relevant datastructures.
    1) Features:
        A [# faces, 12] sized tensor with 12 face-based features. The features are:
        - 3 acute angles of the triangular face               : a0, a1, a2
        - 3 acute angles of pole0                             : a3, a4, a5
        - 3 acute angles of pole1                             : a6, a7, a8
        - 1 triangle-pole0 distance squared over triangle area: p0
        - 1 triangle-pole1 distance squared over triangle area: p1
        - 1 area of the triangle                              : area
        They are arranged as a 1 by 12 vector per face in the order above. For surface
        meshes, a6, a7, a8 are 0 and p1 is 0.

    2) Adjacency:
        A [# faces, 7] sized tensor with values denoting indices of adjacent faces.
        Each row denotes the idx of the face into the features array and the indices
        of its 6 neighbours. For surface faces, the last 3 neighbours are -1.
    :param mesh_file: .msh file containing the tet-mesh (ex: 00001.msh)
    :param output_folder: file in which to write the two datastructures (ex: 0001f.msh, 00001a.msh)
    :return:
    """
    start = time.time()
    tri_idx_map = {}
    idx_tri_map = {}

    def str_to_idx(tri_str):
        if tri_str == "x":
            return -1

        t = str_to_tri(tri_str)
        for p in set(permutations(t)):
            tri_str = tri_to_str(p)
            if tri_str in tri_idx_map:
                return tri_idx_map[tri_str]

    def idx_to_str(ix):
        return idx_tri_map[ix]

    # Read .msh file to get verts and tets
    verts, tets = igl.read_msh(mesh_file)

    # Indices into each tet to get 4 composite triangles
    i0 = [0, 1, 2]
    i1 = [0, 1, 3]
    i2 = [0, 2, 3]
    i3 = [1, 2, 3]
    poles = [3, 2, 1, 0]

    # Track triangles' vertices and angles
    tri_seen = set()
    tri_features = []
    tri_adjacency = []

    for i in range(len(tets)):
        tet = tets[i]
        tris = [tet[i0], tet[i1], tet[i2], tet[i3]]

        for j in range(len(tris)):
            tri = tris[j]

            seen_before = False
            tri_string = ""
            for perm in set(permutations(tri)):
                tri_string = tri_to_str(perm)
                if tri_string in tri_seen:
                    seen_before = True
                    break

            if not seen_before:
                # Calculate ordering of vertices and angles in tri
                v0, v1, v2, a0, a1, a2 = calc_ordered_angles(tri, verts)
                # Calculate pole distance
                pole = tet[poles[j]]
                pole_dist = calc_point_tri_distance(pole, tri, verts)

                # Calculate triangle area
                area = calc_tri_area(verts[v0], verts[v1], verts[v2])

                # Calculate pole distance feature
                pole_dist_feature = pole_dist ** 2 / area

                # Calculate pole angles
                a3 = calc_angle_in_tri(verts[pole], verts[v0], verts[v1])
                a4 = calc_angle_in_tri(verts[pole], verts[v1], verts[v2])
                a5 = calc_angle_in_tri(verts[pole], verts[v2], verts[v0])

                # Set tri_string to idx maps
                idx = len(tri_features)
                tri_string = tri_to_str([v0, v1, v2])
                tri_idx_map[tri_string] = idx
                idx_tri_map[idx] = tri_string

                # Update tri_features
                tri_features.append([a0, a1, a2,
                                     a3, a4, a5,
                                     0, 0, 0,
                                     pole_dist_feature, 0, area])

                # Update tri_adjacency
                tri_adjacency.append([
                    tri_to_str([v0, v1, v2]),
                    tri_to_str([pole, v0, v1]),
                    tri_to_str([pole, v1, v2]),
                    tri_to_str([pole, v2, v0]),
                    "x",
                    "x",
                    "x"
                ])

                # Add to seen
                tri_seen.add(tri_string)
            else:
                # Get ordered triangle vertices
                v0, v1, v2 = str_to_tri(tri_string)

                # Calculate pole distance
                pole = tet[poles[j]]
                pole_dist = calc_point_tri_distance(pole, [v0, v1, v2], verts)

                # Calculate pole angles
                a3 = calc_angle_in_tri(verts[pole], verts[v0], verts[v1])
                a4 = calc_angle_in_tri(verts[pole], verts[v1], verts[v2])
                a5 = calc_angle_in_tri(verts[pole], verts[v2], verts[v0])

                # Get already calculated features
                idx = tri_idx_map[tri_string]

                area = tri_features[idx][11]
                pole_dist_feature0 = tri_features[idx][9]
                pole_dist_feature1 = pole_dist ** 2 / area

                # Check if we need to swap ordering
                if pole_dist_feature1 > pole_dist_feature0:
                    # Need to swap
                    tri_features[idx][6] = tri_features[idx][3]
                    tri_features[idx][7] = tri_features[idx][4]
                    tri_features[idx][8] = tri_features[idx][5]
                    tri_features[idx][3] = a3
                    tri_features[idx][4] = a4
                    tri_features[idx][5] = a5
                    tri_features[idx][9] = pole_dist_feature1
                    tri_features[idx][10] = pole_dist_feature0

                    tri_adjacency[idx][4] = tri_adjacency[idx][1]
                    tri_adjacency[idx][5] = tri_adjacency[idx][2]
                    tri_adjacency[idx][6] = tri_adjacency[idx][3]
                    tri_adjacency[idx][1] = tri_to_str([pole, v0, v1])
                    tri_adjacency[idx][2] = tri_to_str([pole, v1, v2])
                    tri_adjacency[idx][3] = tri_to_str([pole, v2, v0])

                else:
                    # Just need to set zero values
                    tri_features[idx][6] = a3
                    tri_features[idx][7] = a4
                    tri_features[idx][8] = a5
                    tri_features[idx][10] = pole_dist_feature1

                    tri_adjacency[idx][4] = tri_to_str([pole, v0, v1])
                    tri_adjacency[idx][5] = tri_to_str([pole, v1, v2])
                    tri_adjacency[idx][6] = tri_to_str([pole, v2, v0])

    # Create idx based adjacency
    tri_adjacency_idx = [
        [str_to_idx(tri_adjacency[i][j]) for j in range(len(tri_adjacency[0]))]
        for i in range(len(tri_adjacency))
    ]

    # Save adjacency and features as tensors
    features = torch.FloatTensor(tri_features)
    adjacency = torch.IntTensor(tri_adjacency_idx)

    mesh_id = mesh_file[-8:-4]
    torch.save(features, output_folder + "/" + mesh_id + "f.pt")
    torch.save(adjacency, output_folder + "/" + mesh_id + "a.pt")

    end = time.time()
    print("Mesh: " + str(mesh_id) +
          " Features:" + str(features.shape) +
          " Adjacency:" + str(adjacency.shape) +
          " Time:" + str(end-start)[0:6])

    # Sanity checks
    '''
    print(len(tri_seen))
    print(np.array(tri_features).shape)
    print(np.array(tri_adjacency).shape)
    print(np.array(tri_adjacency_idx).shape)
    print(tri_adjacency[0:10])
    print(tri_adjacency_idx[0:10])
    def check(tri_str):
        if tri_str == "x":
            return 1
        s = 0
        t = str_to_tri(tri_str)
        for p in set(permutations(t)):
            tri_str = tri_to_str(p)
            if tri_str in tri_idx_map:
                s += 1
        if s > 1:
            print(tri_str)
        return s
    checker = [
        [check(tri_adjacency[i][j]) for j in range(len(tri_adjacency[0]))]
        for i in range(len(tri_adjacency))
    ]
    print(np.sum(checker) / 7)
    '''

def assemble_example(feature_file, adjacency_file, padded_output_folder, max_faces = 9000):
    """
    Assembles a fully processed mesh example from the features and adjacency datastructures
    defined above. Creates a [max_faces, 7, 12] tensor, pads max_faces - #faces zeros to
    the tensor. For meshes with more than max_faces faces, does nothing.

    :param feature_file: file containing the features datastructure.
    :param adjacency_file: file containing the adjacency datastructure.
    :param padded_output_folder: folder to write output.
    :param max_faces: How many faces to pad each tensor to.
    :return:
    """
    start = time.time()

    features = torch.load(feature_file)
    num_faces = features.shape[0]

    # Disregard meshes with greater than max_faces number of faces
    if num_faces > max_faces:
        return

    adjacencies = torch.load(adjacency_file)
    mesh_idx = feature_file[-9:-4]
    padded_features = torch.zeros(max_faces, 7, 12)
    padded_adjacencies = -torch.ones(max_faces, 7)

    # For each face assemble neighbours features
    for row_idx in range(num_faces):
        adjacency = adjacencies[row_idx]

        neighbourhood = []
        for tri_idx in adjacency:
            tri_idx = tri_idx.numpy()
            if tri_idx == -1:
                neighbourhood.append(np.array([0] * 12))
            else:
                neighbourhood.append(features[tri_idx].numpy())

        neighbourhood = torch.Tensor(neighbourhood)
        neighbourhood = torch.unsqueeze(neighbourhood, dim=0)
        padded_features[row_idx] = neighbourhood

    # Pad adjacencies
    padded_adjacencies[0:num_faces, :] = adjacencies

    # save padded example
    torch.save(padded_features, padded_output_folder + "/" + str(mesh_idx) + ".pt")


    # save padded adjacencies
    torch.save(padded_adjacencies, padded_output_folder + "/" + str(mesh_idx) + "a.pt")

    end = time.time()
    print("Padded mesh " + str(mesh_idx) +
          " with " + str(max_faces - num_faces) +
          " faces in " + str(end - start)[0:6])

def pad_adjacencies():
    max_faces = 9000
    output_folder = "./data/processed_data"
    padded_folder = "./data/padded_data"
    adjacency_files = [join(output_folder, f) for f in listdir(output_folder)
                       if (isfile(join(output_folder, f)) and f[-4:] == "a.pt")]

    for adjacency_file in adjacency_files:
        adjacency = torch.load(adjacency_file)
        num_faces = adjacency.shape[0]
        if num_faces > max_faces:
            continue
        padded_adjacency = -torch.ones(max_faces, 7)
        padded_adjacency[0:num_faces, :] = adjacency
        mesh_idx = adjacency_file[-9:-4]
        torch.save(padded_adjacency, padded_folder + "/" + str(mesh_idx) + "a.pt")
        print(mesh_idx)

# Calculates distance from triangle's plane to a point
def calc_point_tri_distance(point, tri, verts):
    p = verts[point]
    v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
    t1 = v1 - v0
    t2 = v2 - v0
    p = p - v0
    n = np.cross(t1, t2)
    n /= np.linalg.norm(n)

    return abs(np.dot(n, p))

# Calculates angles of triangle tri, and returns vertices and angles
# in order of largest to smallest angle.
def calc_ordered_angles(tri, verts):
    v0 = tri[0]
    v1 = tri[1]
    v2 = tri[2]
    a0 = calc_angle_in_tri(verts[v0], verts[v1], verts[v2])
    a1 = calc_angle_in_tri(verts[v1], verts[v0], verts[v2])
    a2 = calc_angle_in_tri(verts[v2], verts[v0], verts[v1])

    ordering = np.argsort(np.array([a0, a1, a2]))[::-1]

    v0, v1, v2 = np.array([v0, v1, v2])[ordering]
    a0, a1, a2 = np.array([a0, a1, a2])[ordering]

    return v0, v1, v2, a0, a1, a2

def calc_angle_in_tri(a, b, c):
    """
    Calculates angle corresponding to vertex a in the triangle in radians.
    From https://www.geeksforgeeks.org/find-all-angles-of-a-triangle-in-3d/.
    :param a: vertex1 (x, y, z)
    :param b: vertex2 (x, y, z)
    :param c: vertex3 (x, y, z)
    :return: angle_a
    """
    x1, x2, x3, y1, y2, y3, z1, z2, z3 = a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]
    num = (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) + (z2 - z1) * (z3 - z1)
    den = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) * \
          math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2 + (z3 - z1) ** 2)

    return math.acos(num / den)

def calc_tri_area(a, b, c):
    """
    Calculates the area of the triangle from the position of its vertices.
    :param a: vertex1 (x, y, z)
    :param b: vertex2 (x, y, z)
    :param c: vertex3 (x, y, z)
    :return: float area of the triangle.
    """
    a_ = dist(a, b)
    b_ = dist(a, c)
    c_ = dist(b, c)

    return 0.25 * math.sqrt((a_ + b_ + c_)*(-a_ + b_ + c_)*(a_ - b_ + c_)*(a_ + b_ - c_))

def dist(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    z = a[2] - b[2]
    return math.sqrt(x*x + y*y + z*z)

data_folder = "./data/Thingi10k/raw"
output_folder = "./data/Thingi10k/processed_data"
padded_folder = "./data/Thingi10k/padded_data"
meshes = [join(data_folder, f) for f in listdir(data_folder) if (isfile(join(data_folder, f)) and f[-4:] == ".msh")]

process = False
assemble = True

if process:
    for mesh in meshes:
        process_mesh(mesh, output_folder)

faces = []
if assemble:
    feature_files = [join(output_folder, f) for f in listdir(output_folder)
                        if (isfile(join(output_folder, f)) and f[-4:] == "f.pt")]
    adjacency_files = [join(output_folder, f) for f in listdir(output_folder)
                        if (isfile(join(output_folder, f)) and f[-4:] == "a.pt")]

    for i in range(len(feature_files)):
        feature_file = feature_files[i]
        adjacency_file = adjacency_files[i]

        #features = torch.load(feature_file)
        #faces.append(features.shape[0])
        assemble_example(feature_file, adjacency_file, padded_folder, max_faces=14000)

#print(max(faces))
#plt.hist(faces)
#plt.show()