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

def filter_preprocessed_meshes(output_folder, max_num_faces = 9000):
    """
    Takes the output folder from preprocess_meshes() and performs the following operations.
    1) Removes all meshes with more than the maximum number of faces.
    2) Pads all meshes to be (max_num_faces by 7 by 11)
    3) TODO Normalizes per each of the 11 features per mesh
    :param output_folder: string path to output folder containing preprocessed meshes
    :param max_num_faces: int, maximum number of faces per mesh
    :return: None
    """
    processed_meshes = [join(output_folder, f) for f in listdir(output_folder)
                        if (isfile(join(output_folder, f)) and f[-3:] == ".pt")]

    for processed_mesh in processed_meshes:
        tensor = torch.load(processed_mesh)
        num_faces = tensor.shape[0]
        print("Padding mesh with " + str(num_faces) + " faces.")
        if num_faces > 9000:
            # Mark mesh as too large
            os.rename(processed_mesh, processed_mesh + ".x")

        else:
            # Pad mesh and resave it
            res = torch.zeros(max_num_faces, 7, 11)
            res[0:num_faces, :, :] = tensor
            os.remove(processed_mesh)
            torch.save(res, processed_mesh)

def preprocess_meshes(meshes, output_folder):
    """
    Takes a list of .msh files and for each mesh calculates its features and saves its
    (# faces by 7 by 11) local patch features tensor to an output folder. Assumes .msh files are
    named "XXXXX.msh" where X is in [0-9].
    :param meshes: list of .msh files to process
    :param output_folder: path to folder where processed mesh tensors are saved
    :return: none
    """

    for mesh in meshes:
        start = time.time()
        mesh_id = mesh[-9:-4]

        print("Processing mesh " + mesh_id)
        verts, tets = igl.read_msh(mesh)
        tris, tri_tet_dict = get_triangles_and_index(tets)
        features = calculate_element_features(tris, verts, tets, tri_tet_dict)
        end = time.time()

        features_tensor = torch.from_numpy(features)
        torch.save(features_tensor, output_folder + "\\" + mesh_id + ".pt")

        print("Processing time for mesh " + mesh_id + ": " + str(end - start))
        print("Output shape for mesh " + mesh_id + ": " + str(features.shape))
        print("---------------------------------------------------------")


def calculate_element_features(tris, verts, tets, tri_tet_dict):
    """
    Each element is defined as a triangle face and its two conjoining tets. The vertices farthest
    from the triangular face are "poles".

    CALCULATE FEATURES PER FACE:
    Given the tris, verts and tets in a mesh, calculate the following features per element:
    1) The 3 acute angles of the triangle itself (3 in total):
        Order by angle size, these vertices are numbered 1, 2, 3 respectively.
    2) The 3 acute angles of each pole (6 in total)
        Order first by farthest pole from vertex 1 and then by angle towards 1-2 edge, 2-3 edge and then 3-1 edge.
    3) The ratio of the perpendicular distance from each pole to the triangle face squared over the area
       of the triangle. (2 in total)
        Order by the farthest pole from vertex 1.
    Edge faces are padded with zeros.

    GROUP FACES INTO LOCAL PATCHES:
    Then group elements by their local patches (1 central face and 6 neighbouring faces).
        Order by 1-2-p0, 2-3-p0, 3-1-p0, 1-2-p1, 2-3-p1, 3-1-p1

    :param tris: numpy ndarray of size (# tris, 3) containing all triangles in the mesh.
    :param verts: numpy ndarray of size (# verts, 3) containing all vertices in the mesh.
    :param tets: numpy ndarray of size (# tets, 4) containing all tets in the mesh.
    :param tets: dictionary of format {tri_id: [tet_idx0, tet_idx1]} - a triangle id and its two coincident tets
    :return: A size (# faces by # neighbours by 11) numpy array containing the features per each element.
    """
    all_features = {}
    all_neighbours = {}

    # Every triangle in the mesh represents an element
    for tri_id in tri_tet_dict:
        # Get triangle vertex indices
        tri_vert_idxs = np.array(id_to_triangle(tri_id))

        # Get pole vertex indices
        tet0_idx = tri_tet_dict[tri_id][0]
        tet1_idx = tri_tet_dict[tri_id][1]
        pole0_idx = -1
        pole1_idx = -1

        for vert_idx in tets[tet0_idx]:
            if vert_idx in tri_vert_idxs:
                continue
            pole0_idx = vert_idx
        if tet1_idx != -1:
            for vert_idx in tets[tet1_idx]:
                if vert_idx in tri_vert_idxs:
                    continue
                pole1_idx = vert_idx

        # Get triangle vertices
        tri_verts = np.array([verts[tri_vert_idxs[0]], verts[tri_vert_idxs[1]], verts[tri_vert_idxs[2]]])

        # Calculate triangle face's angles
        tri_angles = np.array(tri_calc_angles(tri_verts[0], tri_verts[1], tri_verts[2]))

        # Get ordering by angle magnitudes
        ordering = np.argsort(tri_angles)

        # Order triangle vertices and indexes
        ordered_tri_verts = tri_verts[ordering]
        ordered_tri_verts_idxs = tri_vert_idxs[ordering]
        ordered_tri_angles = tri_angles[ordering]

        v0, v1, v2 = ordered_tri_verts
        v0_idx, v1_idx, v2_idx = ordered_tri_verts_idxs

        # Get pole vertices and order them
        pole0, pole1 = -1, -1
        pole0_dist_sqr, pole1_dist_sqr = 0, 0
        if pole1_idx == -1:
            pole0 = verts[pole0_idx]
        else:
            pole0 = verts[pole0_idx]
            pole1 = verts[pole1_idx]
            pole0_dist_sqr = dist_sqr(pole0, v0)
            pole1_dist_sqr = dist_sqr(pole1, v0)

            if pole1_dist_sqr > pole0_dist_sqr:
                pole0_idx, pole1_idx = pole1_idx, pole0_idx
                pole0, pole1 = pole1, pole0
                pole0_dist_sqr, pole1_dist_sqr = pole1_dist_sqr, pole0_dist_sqr

        # Calculate pole angles
        ordered_pole_angles = np.zeros(6)
        if pole1_idx == -1: # edge face
            ordered_pole_angles[0] = tri_calc_angle(pole0, v0, v1)
            ordered_pole_angles[1] = tri_calc_angle(pole0, v1, v2)
            ordered_pole_angles[2] = tri_calc_angle(pole0, v2, v0)
        else:
            ordered_pole_angles[0] = tri_calc_angle(pole0, v0, v1)
            ordered_pole_angles[1] = tri_calc_angle(pole0, v1, v2)
            ordered_pole_angles[2] = tri_calc_angle(pole0, v2, v0)
            ordered_pole_angles[3] = tri_calc_angle(pole1, v0, v1)
            ordered_pole_angles[4] = tri_calc_angle(pole1, v1, v2)
            ordered_pole_angles[5] = tri_calc_angle(pole1, v2, v0)

        # Calculate pole length ratios
        tri_area = tri_calc_area(v0, v1, v2)
        r0 = pole0_dist_sqr / tri_area
        r1 = pole1_dist_sqr / tri_area

        # Assemble feature vector for this face
        features = np.hstack([ordered_tri_angles, ordered_pole_angles, r0, r1])
        all_features[tri_id] = features

        # Get neighbours in local patch for this face
        t0 = [v0_idx, v1_idx, v2_idx]
        t1 = [v0_idx, v1_idx, pole0_idx]
        t2 = [v1_idx, v2_idx, pole0_idx]
        t3 = [v2_idx, v0_idx, pole0_idx]
        t4 = -1 # Nonexistent neighbouring faces for a face on the edge of the mesh
        t5 = -1
        t6 = -1
        if pole1_idx != -1:
            t4 = [v0_idx, v1_idx, pole1_idx]
            t5 = [v1_idx, v2_idx, pole1_idx]
            t6 = [v2_idx, v0_idx, pole1_idx]
        ordered_neighbours = [t0, t1, t2, t3, t4, t5, t6]
        all_neighbours[tri_id] = ordered_neighbours

    # Use all_features and all_neighbours to generate result
    res = []
    for tri_id in all_neighbours:
        patch = all_neighbours[tri_id]

        patch_features = []
        for tri in patch:
            if tri != -1:
                for perm in set(permutations(tri)):
                    _tri_id = triangle_to_id(perm)
                    if _tri_id in all_features:
                        patch_features.append(all_features[_tri_id])
                        break
            else:
                patch_features.append([0]*11) # Pad with zeros

        res.append(patch_features)

    return np.array(res)

def tri_calc_angles(a, b, c):
    """
    Calculates all angles of the triangle (a, b, c)
    :param a: vertex1 (x, y, z)
    :param b: vertex2 (x, y, z)
    :param c: vertex3 (x, y, z)
    :return: angle_a, angle_b, angle_c
    """

    return tri_calc_angle(a, b, c), tri_calc_angle(b, a, c), tri_calc_angle(c, a, b)

def tri_calc_angle(a, b, c):
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

def tri_calc_area(a, b, c):
    """
    Calculates the area of the triangle from the position of its vertices.
    :param a: vertex1 (x, y, z)
    :param b: vertex2 (x, y, z)
    :param c: vertex3 (x, y, z)
    :return: float area of the triangle.
    """
    l1 = dist(a, b)
    l2 = dist(a, c)
    l3 = dist(b, c)
    p = (l1 + l2 + l3) / 2

    return math.sqrt(p * (p-l1) * (p-l2) * (p-l3))

def get_triangles_and_index(tets):
    """
    Given the tets in a tet mesh, returns a list of all triangles in the mesh.
    Additionaly creates a dictionary that indexes each triangle with its two coincident tets. Tri's on the surface
    of the mesh will have -1 to indicate no second tet, ie: {tri_id: [tet_idx0, -1]}
    :param tets: numpy ndarray of size (# tets, 4) containing all tets in the mesh.
    :return: numpy ndarray of size (# triangles, 3) and dictionary of form {tri_id: [tet_idx0, tet_idx1]} for all tris
    """
    # String representations of final triangles, ie: (1, 2, 3) = "1.2.3"
    tri_ids = []
    # Dictionary will contain 2 tets for each triangle
    tri_tet_dict = {}

    # Indices into each tet to get 4 composite triangles
    i0 = [0, 1, 2]
    i1 = [0, 2, 3]
    i2 = [0, 3, 1]
    i3 = [3, 2, 1]

    for tet_idx in range(len(tets)):
        tet = tets[tet_idx]

        # Get 4 composite triangles from the tet
        tris = [tet[i0], tet[i1], tet[i2], tet[i3]]

        for tri in tris:
            # Check that this triangle has not already been seen
            new_tri = True
            for perm in set(permutations((tri))):
                tri_id = triangle_to_id(perm)
                if tri_id in tri_ids:
                    new_tri = False
                    break

            if new_tri:
                tri_ids.append(tri_id)
                tri_tet_dict[tri_id] = [tet_idx, -1]
            else:
                tri_tet_dict[tri_id][1] = tet_idx

    # Convert triangle ids back to triangles
    tris = [id_to_triangle(tri_id) for tri_id in tri_ids]

    return np.array(tris), tri_tet_dict

def triangle_to_id(tri):
    return "{}.{}.{}".format(tri[0], tri[1], tri[2])

def id_to_triangle(id):
    return [int(num) for num in re.findall('(\d+)\.(\d+)\.(\d+)', id)[0]]

def dist_sqr(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    z = a[2] - b[2]
    return x*x + y*y + z*z

def dist(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    z = a[2] - b[2]
    return math.sqrt(x*x + y*y + z*z)

def plot_vertices(points):
    fig = figure()
    ax = Axes3D(fig)

    points = np.vstack([points, points[0]])
    ax.plot(points[:,0], points[:,1], points[:,2])

    for i in range(len(points)-1):
        ax.text(points[i, 0], points[i, 1], points[i, 2], '%s' % (str(i)), size=20, zorder=1, color='k')

    plt.show()

def plot_histogram_num_faces(processed_data_folder):
    num_faces = []
    cached_sizes_file = join(processed_data_folder, "sizes.pt")
    if isfile(cached_sizes_file):
        num_faces = torch.load(cached_sizes_file).tolist()
    else:
        tensor_files = [join(processed_data_folder, f) for f in listdir(processed_data_folder)
                        if (isfile(join(processed_data_folder, f)) and f[-3:] == ".pt")]
        num_faces = [torch.load(f).shape[0] for f in tensor_files]
        sizes = torch.from_numpy(np.array(num_faces))
        torch.save(sizes, processed_data_folder + "\\" + "sizes.pt")

    fig, hist_plot = plt.subplots(1, 1)
    hist_plot.hist(num_faces, 20)
    hist_plot.set_ylabel("Frequency")
    hist_plot.set_xlabel("Number of Faces")
    hist_plot.set_title("Histogram of Number of Mesh Faces")

    plt.show()

if __name__ == "__main__":
    # Demo for a single mesh
    if False:
        # Read in mesh
        verts, tets = igl.read_msh("./data/00001.msh")

        # Find all triangles in the mesh and index tets by these triangles
        tris, tri_tet_dict = get_triangles_and_index(tets)

        # Calculate features defined per triangle face in the mesh
        start = time.time()
        features = calculate_element_features(tris, verts, tets, tri_tet_dict)
        end = time.time()
        print("Processing time for mesh 0:")
        print(end-start)
        print("Output shape for mesh 0:")
        print(features.shape)

    # Initally process all .msh files in the data_folder and save all processed meshes in the output_folder
    data_folder = '.\\data'
    output_folder = '.\\data\\processed_data'
    if False:
        # Get all .msh files
        meshes = [join(data_folder, f) for f in listdir(data_folder) if (isfile(join(data_folder, f)) and f[-4:] == ".msh")]

        # Preprocess all .msh files into (# faces by 7 by 11) patch tensors and saves them
        preprocess_meshes(meshes, output_folder)

    # Histogram to see distribution of # of faces in all meshes
    # plot_histogram_num_faces(output_folder)

    # Filters tensors by:
    # 1) Removing all meshs with more than 9,000 faces
    # 2) Padding all meshes to be (9,000 by 7 by 11)
    # 3) Normalize the mesh features.
    filter_preprocessed_meshes(output_folder)
