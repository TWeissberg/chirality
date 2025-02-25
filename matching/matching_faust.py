import os
import glob
from pathlib import Path
#from torch_cluster import knn
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors
import networkx as nx
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
#from dataloaders.mesh_container import MeshContainer
import pandas as pd
import torch
import numpy as np
import numbers
import scipy.io as sio
from plyfile import PlyData
import torchvision
import torch.nn as nn
import sys
import math

def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy()"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, numbers.Number):
        return np.array(tensor)
    else:
        raise NotImplementedError



class MeshContainer(object):
    """
    Helper class to store face, vert as numpy

    Control I/O, convertions and other mesh utilities
    """
    def __init__(self, vert=None, face=None):
        if (vert is None and face is None): return
        if (not type(vert).__module__ == np.__name__):
            vert = vert.cpu().detach().numpy()
        if (face is not None and not type(face).__module__ == np.__name__):
            face = face.cpu().detach().numpy().astype(np.int32)
        if (face is not None and face.shape[0] == 3): face = face.transpose()

        self.vert, self.face = vert, face
        self.n = self.vert.shape[0]

    def copy(self):
        return MeshContainer(self.vert.copy(), self.face.copy())

    def _load_from_file_mat(self, file_path, dataset='faust_reg'):
        self.load_file_name = file_path
        mesh_mat = sio.loadmat(file_path)

        if (dataset == 'tosca'):
            vx = mesh_mat['surface']['X'][0, 0]
            vy = mesh_mat['surface']['Y'][0, 0]
            vz = mesh_mat['surface']['Z'][0, 0]
            face = np.array(mesh_mat['surface']['TRIV'][0, 0].astype(np.int32))
        elif ('null' in file_path
              or ('partial' in file_path and 'rescaled' not in file_path)):
            points = mesh_mat['N']['xyz'][0][0]
            vx = points[:, 0][:, None]
            vy = points[:, 1][:, None]
            vz = points[:, 2][:, None]
            face = np.array(mesh_mat['N']['tri'][0][0].astype(np.int32))
        elif ('model' in file_path and 'remesh' in file_path):
            vx = mesh_mat['part']['X'][0, 0]
            vy = mesh_mat['part']['Y'][0, 0]
            vz = mesh_mat['part']['Z'][0, 0]
            face = np.array(mesh_mat['part']['triv'][0, 0].astype(np.int32))
        elif ('remesh' in file_path and 'deform' not in file_path):
            vx = mesh_mat['model_remesh']['X'][0, 0]
            vy = mesh_mat['model_remesh']['Y'][0, 0]
            vz = mesh_mat['model_remesh']['Z'][0, 0]
            face = np.array(mesh_mat['model_remesh']['triv'][0, 0].astype(
                np.int32))
        else :
            vx = mesh_mat['VERT'][:, 0][:, None]
            vy = mesh_mat['VERT'][:, 1][:, None]
            vz = mesh_mat['VERT'][:, 2][:, None]
            face = np.array(mesh_mat['TRIV'].astype(np.int32))

        if np.min([face]) > 0:
            face = face - 1
        self.face = face
        self.vert = np.concatenate((vx, vy, vz), axis=1)
        self.n = self.vert.shape[0]

    def _load_from_file_ply(self, file_path):
        self.load_file_name = file_path
        ply_data = PlyData.read(file_path)
        vx = np.array(ply_data['vertex'].data['x'])[:, np.newaxis]
        vy = np.array(ply_data['vertex'].data['y'])[:, np.newaxis]
        vz = np.array(ply_data['vertex'].data['z'])[:, np.newaxis]
        self.vert = np.concatenate((vx, vy, vz), axis=1)
        self.face = ply_data['face'].data['vertex_indices'][:]
        if (len(self.face.shape) < 2):
            self.face = np.array([x for x in self.face])
        self.n = self.vert.shape[0]

    def _load_from_file_obj(self, file_path):
        try:
            # with open(file_path, 'r') as f:
            #     vertices = []
            #     faces = []
            #     for line in f:
            #         line = line.strip()
            #         if line == '' or line[0] == '#':
            #             continue
            #         line = line.split()
            #         if line[0] == 'v':
            #             vertices.append([float(x) for x in line[1:]])
            #         elif line[0] == 'f':
            #             faces.append([int(x.split('/')[0]) - 1 for x in line[1:]])
            # self.vert, self.face = np.asarray(vertices), np.asarray(faces)
            # self.n = self.vert.shape[0]
            raise Exception()
        except:
            import trimesh
            mesh = trimesh.exchange.obj.load_obj(open(file_path,'rb'))
            self.vert = mesh['vertices']
            self.face = mesh['faces']
            self.n = self.vert.shape[0]
            self.assignment = mesh["vertex_colors"][:, 2].astype(np.int32)


        # import trimesh
        # mesh = trimesh.exchange.obj.load_obj(open(file_path,'rb'))
        # self.vert = mesh['vertices']
        # self.face = mesh['faces']
        # self.n = self.vert.shape[0]
        # return
        # file = open(file_path, 'r+')
        # os.system('meshlabserver -i ' + file_path + ' -o ' +
        #           os.path.basename(file_path)[:-4] + '.ply')
        # return self._load_from_file_ply(
        #     os.path.basename(file_path)[:-4] + '.ply')
    
    def _load_from_file_off(self, file_path):
        import trimesh
        mesh = trimesh.exchange.off.load_off(open(file_path,'rb'))
        self.vert = mesh['vertices']
        self.face = mesh['faces']
        self.n = self.vert.shape[0]


    def load_from_file(self, file_path, dataset=''):
        self.file_path = file_path
        filename = os.path.basename(file_path)
        if filename.endswith('mat'):
            self._load_from_file_mat(file_path, dataset)
        elif filename.endswith('ply'):
            self._load_from_file_ply(file_path)
        elif filename.endswith('obj'):
            self._load_from_file_obj(file_path)
        elif filename.endswith('off'):
            self._load_from_file_off(file_path)
        elif (filename.isnumeric()):
            self = self._load_from_raw(file_path)
        return self

    def save_to_ply_and_obj(self):
        vertex = np.array([tuple(i) for i in self.vert],
                          dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        face = np.array(
            [(tuple(i), ) for i in self.face],
            dtype=[("vertex_indices", "i4", (3, ))],
        )
        el = PlyElement.describe(vertex, "vertex")
        el2 = PlyElement.describe(face, "face")
        plydata = PlyData([el, el2])
        plydata.write(self.load_file_name[:-4] + '.ply')

        import trimesh
        mesh1 = trimesh.load(self.load_file_name[:-4] + '.ply')
        with open(self.load_file_name[:-4] + '.obj', "w") as text_file:
            text_file.write(trimesh.exchange.obj.export_obj(mesh1))

    def save_as_mat(self, file_path=''):
        face_to_save = self.face if (np.min(self.face) > 0) else (self.face +
                                                                  1)
        mesh = {'face': face_to_save.tolist(), 'VERT': self.vert.tolist()}
        if (file_path == ''):
            file_path = self.file_path
        sio.savemat(file_path[:-4] + '.mat', mesh)

    def get_area_of_faces(self):
        """
        Compute the areas of all triangles on the mesh.
        Parameters
        ----------
        Returns
        -------
        area: 1-D numpy array
            area[i] is the area of the i-th triangle
        """
        areas = np.zeros(self.face.shape[0])

        for i, triangle in enumerate(self.face):
            a = np.linalg.norm(self.vert[triangle[0]] - self.vert[triangle[1]])
            b = np.linalg.norm(self.vert[triangle[1]] - self.vert[triangle[2]])
            c = np.linalg.norm(self.vert[triangle[2]] - self.vert[triangle[0]])
            s = (a + b + c) / 2.0
            areas[i] = np.sqrt(s * (s - a) * (s - b) * (s - c))
        return areas

    def compute_LBO(self, num_evecs=15):
        spectrum_results = fem_laplacian(self.vert,
                                         self.face,
                                         num_evecs,
                                         normalization="areaindex",
                                         areas=self.get_area_of_faces())
        self_functions = spectrum_results['self_functions'].copy()
        return self_functions




def compute_geodesic_distmat(verts, faces):
    """
    Compute geodesic distance matrix using Dijkstra algorithm

    Args:
        verts (np.ndarray): array of vertices coordinates [n, 3]
        faces (np.ndarray): array of triangular faces [m, 3]

    Returns:
        geo_dist: geodesic distance matrix [n, n]
    """
    NN = 500

    # get adjacency matrix
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertex_adjacency = mesh.vertex_adjacency_graph
    assert nx.is_connected(vertex_adjacency), 'Graph not connected'
    vertex_adjacency_matrix = nx.adjacency_matrix(vertex_adjacency, range(verts.shape[0]))
    # get adjacency distance matrix
    graph_x_csr = neighbors.kneighbors_graph(verts, n_neighbors=NN, mode='distance', include_self=False)
    distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
    distance_adj[vertex_adjacency_matrix != 0] = graph_x_csr[vertex_adjacency_matrix != 0]
    # compute geodesic matrix
    geodesic_x = shortest_path(distance_adj, directed=False)
    if np.any(np.isinf(geodesic_x)):
        print('Inf number in geodesic distance. Increase NN.')
    return geodesic_x / math.sqrt(mesh.area)



def square_distance(src, dst):
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1, 0))
    dist += torch.sum(src**2, -1).view(N, 1)
    dist += torch.sum(dst**2, -1).view(1, M)
    return dist


def cosine_similarity(a, b):
    if len(a) > 10000:
        return cosine_similarity_batch(a, b, batch_size=30000)
    dot_product = torch.mm(a, b.t())
    norm_a = torch.norm(a, dim=1, keepdim=True)
    norm_b = torch.norm(b, dim=1, keepdim=True)
    similarity = dot_product / (norm_a * norm_b.t())

    return similarity


def cosine_similarity_batch(a, b, batch_size=30000):
    num_a, dim_a = a.size()
    num_b, dim_b = b.size()
    similarity_matrix = torch.empty(num_a, num_b, device="cpu")
    for i in range(0, num_a, batch_size):
        a_batch = a[i:i+batch_size]
        for j in range(0, num_b, batch_size):
            b_batch = b[j:j+batch_size]
            dot_product = torch.mm(a_batch, b_batch.t())
            norm_a = torch.norm(a_batch, dim=1, keepdim=True)
            norm_b = torch.norm(b_batch, dim=1, keepdim=True)
            similarity_batch = dot_product / (norm_a * norm_b.t())
            similarity_matrix[i:i+batch_size, j:j+batch_size] = similarity_batch.cpu()
    return similarity_matrix


def compute_acc(verts1, verts2, geo_matrix, cos_similarity, gt_corres):

    verts1 = verts1[gt_corres>-0.5,:]
    cos_similarity = cos_similarity[gt_corres>-0.5,:]
    gt_corres = gt_corres[gt_corres>-0.5]

    label = matrix_map_from_corr_map(gt_corres, verts1, verts2)

    print("label computation finished!")

    ratio_list = (0.01 * np.arange(0, 101)).tolist()


    track_dict = {}
    corr_tensor = prob_to_corr_test(cos_similarity)
    hit = label.argmax(-1)
    pred_hit = cos_similarity.argmax(-1)
    #target_dist = square_distance(input2, input2)
    track_dict["geo_error"] = geo_matrix[pred_hit, hit].mean()
    acc_000 = label_ACC_percentage_for_inference(corr_tensor, label)
    track_dict["acc_0.00"] = acc_000.item()

    for each_ratio in ratio_list:

        val = make_soft_label(label, verts2, ratio=each_ratio)
        val_geo = make_soft_label_geo(label, verts2, geo_matrix, ratio=each_ratio)
        track_dict["acc_" + str(each_ratio)] = label_ACC_percentage_for_inference(corr_tensor, val).item()
        track_dict["geo_acc_" + str(each_ratio)] = label_ACC_percentage_for_inference(corr_tensor, val_geo).item()

    return track_dict


def matrix_map_from_corr_map(matrix_map, source, target):
    # matrix_map.shape is [N,], output.shape is [N,M]
    matrix_map = matrix_map.clone().detach().to(source.device)
    inputs = torch.ones((source.shape[0], target.shape[0])).to(matrix_map.device)
    outputs = torch.zeros((source.shape[0], target.shape[0])).to(matrix_map.device)

    label = outputs.scatter(dim=1, index=matrix_map.unsqueeze(1).long(), src=inputs)
    return label




# def extract_soft_labels_per_pair(gt_map, verts2, geo_matrix, replace_on_cpu=False):

#     ratio_list = (0.01 * np.arange(0, 101)).tolist()
#     soft_labels = {}
#     soft_labels_geo = {}
#     for each_ratio in ratio_list:
#         val = make_soft_label(gt_map, verts2, ratio=each_ratio)
#         print("soft label finished")
#         if replace_on_cpu:
#             val = val.cpu()
#         soft_labels[f"{each_ratio}"] = val
#         val_geo = make_soft_label_geo(gt_map, verts2, geo_matrix, ratio=each_ratio)
#         print("soft geodesic label finished")
#         if replace_on_cpu:
#             val_geo = val_geo.cpu()
#         soft_labels_geo[f"{each_ratio}"] = val_geo

#     return ratio_list, soft_labels, soft_labels_geo


def make_soft_label(label_origin, xyz2, ratio=0.5):
    if ratio == 0.0:
        return label_origin
    else:
        soft_label = label_origin.clone()

        dist = torch.cdist(xyz2, xyz2) ** 2

        #max_square_radius = torch.max(dist)

        radius = ratio# * torch.sqrt(max_square_radius)

        dists_from_source = dist[soft_label.nonzero(as_tuple=False)[:, 1]]
        mask = dists_from_source <= radius**2
        soft_label[: mask.shape[0]][mask] = 1
        return soft_label

def make_soft_label_geo(label_origin, xyz2, geo_dismat, ratio=0.5):
    #geo_dismat = geo_dismat
    if ratio == 0.0:
        return label_origin
    else:
        soft_label = label_origin.clone()

        #dist = torch.cdist(xyz2, xyz2) ** 2

        max_square_radius = torch.max(geo_dismat)

        radius = ratio * max_square_radius

        dists_from_source = geo_dismat[soft_label.nonzero(as_tuple=False)[:, 1]]
        mask = dists_from_source <= radius
        soft_label[: mask.shape[0]][mask] = 1
        return soft_label



def label_ACC_percentage_for_inference(label_in, label_gt):
    assert label_in.shape == label_gt.shape
    label_in = label_in.cuda()

    element_product = torch.mul(label_in, label_gt.cuda())
    N1 = label_in.shape[0]
    sum_row = torch.sum(element_product, dim=-1)  # N1x1

    hit = (sum_row != 0).sum()
    acc = hit.float() / torch.tensor(N1).float()
    
    return acc * 100.0


def prob_to_corr_test(prob_matrix):
    c = torch.zeros_like(prob_matrix)
    idx = torch.argmax(prob_matrix, dim=1, keepdim=True)
    for each_row in range(c.shape[0]):
        c[each_row][idx[each_row]] = 1.0

    return c



class ChiralityMLP(nn.Module):
    def __init__(self, feature_size, num_layers):
        super(ChiralityMLP, self).__init__()

        if num_layers == 0:
            self.forward_model = torch.nn.Identity()
            self.backward_model = torch.nn.Identity()
        else:
            self.forward_model = torchvision.ops.MLP(feature_size, [feature_size] * num_layers)
            self.backward_model = torchvision.ops.MLP(feature_size, [feature_size] * num_layers)



class ChiralityDisentangler_ICCV(nn.Module):
    def __init__(self, feature_size, num_layers, model_type, normalization, chirality_dim = 1, force_orthogonal = False):
        super(ChiralityDisentangler_ICCV, self).__init__()
        
        assert model_type in ["invertible", "mlp"]
        if model_type == "invertible":
            self.m = ChiralityInvertible(feature_size, num_layers)
        elif model_type == "mlp":
            self.m = ChiralityMLP(feature_size, num_layers)

        if force_orthogonal:
            self.A = nn.utils.parametrizations.orthogonal(nn.Linear(feature_size, feature_size, bias = False), orthogonal_map = "cayley")
        else:
            self.A = nn.Linear(feature_size, feature_size, bias = False)

        assert normalization in ["matrix", "tanh", "before", "beforeAndAfter"]
        self.normalization = normalization

        self.new_feature_dim = chirality_dim
    

    def forward(self, x, return_backward=False):
        x = self.m.forward_model(x)

        forward_features = x.clone()
        
        backward_features = None
        if return_backward:
            backward_features = self.m.backward_model(x)
        
        if self.normalization == "before" or self.normalization == "beforeAndAfter":
            x = torch.nn.functional.normalize(x, dim = -1)

        x = self.A(x)
        
        if self.normalization == "matrix":
            x = torch.nn.functional.normalize(x, dim = -1)
        elif self.normalization == "tanh":
            x = torch.tanh(x)

        chiral, non_chiral = x.split([self.new_feature_dim, x.shape[2] - self.new_feature_dim], -1)

        if self.normalization == "beforeAndAfter":
            non_chiral = torch.nn.functional.normalize(non_chiral, dim = -1)

        if return_backward:
            return chiral, non_chiral, forward_features, backward_features
        else:
            return chiral, non_chiral, forward_features




class ChiralityDisentangler_SIGGRAPH(nn.Module):
    def __init__(self, feature_size, num_layers, model_type, normalization, chirality_dim = 1, force_orthogonal = False):
        super(ChiralityDisentangler_SIGGRAPH, self).__init__()
        
        assert model_type in ["invertible", "mlp"]
        if model_type == "invertible":
            self.m = ChiralityInvertible(feature_size, num_layers)
        elif model_type == "mlp":
            self.m = ChiralityMLP(feature_size, num_layers)

        if force_orthogonal:
            self.A = nn.utils.parametrizations.orthogonal(nn.Linear(feature_size, feature_size, bias = False), orthogonal_map = "cayley")
        else:
            self.A = nn.Linear(feature_size, feature_size, bias = False)

        assert normalization in ["matrix", "tanh", "before", "beforeAndAfter"]
        self.normalization = normalization

        self.new_feature_dim = chirality_dim
    

    def forward(self, x, return_backward=False):
        x = self.m.forward_model(x) + x

        forward_features = x.clone()
        
        backward_features = None
        if return_backward:
            backward_features = self.m.backward_model(x) + x
        
        if self.normalization == "before" or self.normalization == "beforeAndAfter":
            x = torch.nn.functional.normalize(x, dim = -1)

        x = self.A(x)
        
        if self.normalization == "matrix":
            x = torch.nn.functional.normalize(x, dim = -1)
        elif self.normalization == "tanh":
            x = torch.tanh(x)

        chiral, non_chiral = x.split([self.new_feature_dim, x.shape[2] - self.new_feature_dim], -1)

        if self.normalization == "beforeAndAfter":
            non_chiral = torch.nn.functional.normalize(non_chiral, dim = -1)

        if return_backward:
            return chiral, non_chiral, forward_features, backward_features
        else:
            return chiral, non_chiral, forward_features



# def compute_acc(label, ratio_list, soft_labels, soft_labels_geo, p, input2, geo_matrix):
#     track_dict = {}
#     corr_tensor = prob_to_corr_test(p)
#     hit = label.argmax(-1)
#     pred_hit = p.argmax(-1)
#     #target_dist = square_distance(input2, input2)
#     track_dict["geo_error"] = geo_matrix[pred_hit, hit].mean()

#     acc_000 = label_ACC_percentage_for_inference(corr_tensor, label)

#     track_dict["acc_0.00"] = acc_000.item()
#     for idx, ratio in enumerate(ratio_list):
#         soft_label_ratio = soft_labels[f"{ratio}"]
#         soft_label_ratio_geo = soft_labels_geo[f"{ratio}"]
#         track_dict["acc_" + str(ratio)] = label_ACC_percentage_for_inference(corr_tensor, soft_label_ratio).item()
#         track_dict["geo_acc_" + str(ratio)] = label_ACC_percentage_for_inference(corr_tensor, soft_label_ratio_geo).item()

#     return track_dict



data_path = "/lustre/scratch/data/tweissbe_hpc-becos-all/faust_o/test"
model_path_iccv = "/lustre/scratch/data/tweissbe_hpc-becos-all/models/iccv/faust_o/files/model.pt"
model_path_siggraph = "/lustre/scratch/data/tweissbe_hpc-becos-all/models/siggraph/faust_o/files/model.pt"

#data_path = "./test"


device = "cuda"
mode = sys.argv[1]
tracks = []
pairs = []

with torch.no_grad():
    for i in tqdm(os.listdir(data_path)):
        full_path = os.path.join(data_path, i)
        shape_file0 = glob.glob(str(Path(full_path) / "0_*.off"))
        mesh0 = MeshContainer().load_from_file(shape_file0[0])

        shape_file1 = glob.glob(str(Path(full_path) / "1_*.off"))
        mesh1 = MeshContainer().load_from_file(shape_file1[0])

        geodesic_1 = compute_geodesic_distmat(mesh1.vert, mesh1.face)
        #geodesic_1 = torch.rand(mesh1.vert.shape[0],mesh1.vert.shape[0]).to(device=device)
        geodesic_1 = torch.tensor(geodesic_1).to(device=device)

        print("geodesic matrix computation finished!")

        if mode == "diff3f":

            feature0 = torch.load(full_path + "/0_features/diff3f_features.pt").to(device=device)
            feature1 = torch.load(full_path + "/1_features/diff3f_features.pt").to(device=device)

        elif mode == "sd+dino":

            dino_feature0 = torch.load(full_path + "/0_features/textured_dino_features.pt")[0].to(device=device)
            sd_feature0 = torch.load(full_path + "/0_features/textured_sd_features.pt")[0].to(device=device)
            feature0 = torch.cat([0.5*sd_feature0/torch.norm(sd_feature0, dim=-1, keepdim=True),0.5*dino_feature0/torch.norm(dino_feature0, dim=-1, keepdim=True)], dim=-1)
            feature0 = feature0/torch.norm(feature0, dim=-1, keepdim=True)

            dino_feature1 = torch.load(full_path + "/1_features/textured_dino_features.pt")[0].to(device=device)
            sd_feature1 = torch.load(full_path + "/1_features/textured_sd_features.pt")[0].to(device=device)
            feature1 = torch.cat([0.5*sd_feature1/torch.norm(sd_feature1, dim=-1, keepdim=True),0.5*dino_feature1/torch.norm(dino_feature1, dim=-1, keepdim=True)], dim=-1)
            feature1 = feature1/torch.norm(feature1, dim=-1, keepdim=True)


        elif mode == "iccv":

            model = ChiralityDisentangler_ICCV(feature_size=3200+768, num_layers=2, model_type="mlp", normalization="matrix", chirality_dim=1, force_orthogonal=False).to(device)
            model.load_state_dict(torch.load(model_path_iccv, weights_only=True))

            dino_feature0 = torch.load(full_path + "/0_features/textured_dino_features.pt")[0].to(device)
            sd_feature0 = torch.load(full_path + "/0_features/textured_sd_features.pt")[0].to(device)
            combined_feature0 = torch.cat([0.5*sd_feature0/torch.norm(sd_feature0, dim=-1, keepdim=True),0.5*dino_feature0/torch.norm(dino_feature0, dim=-1, keepdim=True)], dim=-1)
            combined_feature0 = combined_feature0/torch.norm(combined_feature0, dim=-1, keepdim=True)

            dino_feature1 = torch.load(full_path + "/1_features/textured_dino_features.pt")[0].to(device)
            sd_feature1 = torch.load(full_path + "/1_features/textured_sd_features.pt")[0].to(device)
            combined_feature1 = torch.cat([0.5*sd_feature1/torch.norm(sd_feature1, dim=-1, keepdim=True),0.5*dino_feature1/torch.norm(dino_feature1, dim=-1, keepdim=True)], dim=-1)
            combined_feature1 = combined_feature1/torch.norm(combined_feature1, dim=-1, keepdim=True)


            diff0 = torch.load(full_path + "/0_features/diff3f_features.pt").to(device=device)
            diff1 = torch.load(full_path + "/1_features/diff3f_features.pt").to(device=device)

            chirality_feature0, _, _ = model(combined_feature0.unsqueeze(0).float())
            chirality_feature1, _, _ = model(combined_feature1.unsqueeze(0).float())


            feature0 = diff0 * chirality_feature0[0]
            feature1 = diff1 * chirality_feature1[0]

        elif mode == "siggraph":

            model = ChiralityDisentangler_SIGGRAPH(feature_size=3200+768, num_layers=2, model_type="mlp", normalization="beforeAndAfter", chirality_dim=1, force_orthogonal=True).to(device)
            model.load_state_dict(torch.load(model_path_siggraph, weights_only=True))

            dino_feature0 = torch.load(full_path + "/0_features/textured_dino_features.pt")[0].to(device)
            sd_feature0 = torch.load(full_path + "/0_features/textured_sd_features.pt")[0].to(device)
            combined_feature0 = torch.cat([0.5*sd_feature0/torch.norm(sd_feature0, dim=-1, keepdim=True),0.5*dino_feature0/torch.norm(dino_feature0, dim=-1, keepdim=True)], dim=-1)
            combined_feature0 = combined_feature0/torch.norm(combined_feature0, dim=-1, keepdim=True)

            dino_feature1 = torch.load(full_path + "/1_features/textured_dino_features.pt")[0].to(device)
            sd_feature1 = torch.load(full_path + "/1_features/textured_sd_features.pt")[0].to(device)
            combined_feature1 = torch.cat([0.5*sd_feature1/torch.norm(sd_feature1, dim=-1, keepdim=True),0.5*dino_feature1/torch.norm(dino_feature1, dim=-1, keepdim=True)], dim=-1)
            combined_feature1 = combined_feature1/torch.norm(combined_feature1, dim=-1, keepdim=True)

            chirality_feature0, non_chirality_feature0, _ = model(combined_feature0.unsqueeze(0).float())
            chirality_feature1, non_chirality_feature1, _ = model(combined_feature1.unsqueeze(0).float())


            difffeature0 = torch.load(full_path + "/0_features/diff3f_features.pt").to(device=device)
            difffeature1 = torch.load(full_path + "/1_features/diff3f_features.pt").to(device=device)


            #feature0 = torch.cat([chirality_feature0[0], non_chirality_feature0[0]], dim=-1)
            #feature1 = torch.cat([chirality_feature1[0], non_chirality_feature1[0]], dim=-1)
            feature0 = torch.cat([difffeature0, chirality_feature0[0], non_chirality_feature0[0]], dim=-1)
            feature1 = torch.cat([difffeature1, chirality_feature1[0], non_chirality_feature1[0]], dim=-1)

        elif mode == "gt_chirality_cat":

            mask_feature0 = torch.tensor(np.loadtxt(full_path + "/0_chirality.txt"))[...,None].to(device=device)
            mask_feature1 = torch.tensor(np.loadtxt(full_path + "/1_chirality.txt"))[...,None].to(device=device)

            model = ChiralityDisentangler_SIGGRAPH(feature_size=3200+768, num_layers=2, model_type="mlp", normalization="beforeAndAfter", chirality_dim=1, force_orthogonal=True).to(device)
            model.load_state_dict(torch.load(model_path_siggraph, weights_only=True))

            dino_feature0 = torch.load(full_path + "/0_features/textured_dino_features.pt")[0].to(device)
            sd_feature0 = torch.load(full_path + "/0_features/textured_sd_features.pt")[0].to(device)
            combined_feature0 = torch.cat([0.5*sd_feature0/torch.norm(sd_feature0, dim=-1, keepdim=True),0.5*dino_feature0/torch.norm(dino_feature0, dim=-1, keepdim=True)], dim=-1)
            combined_feature0 = combined_feature0/torch.norm(combined_feature0, dim=-1, keepdim=True)

            dino_feature1 = torch.load(full_path + "/1_features/textured_dino_features.pt")[0].to(device)
            sd_feature1 = torch.load(full_path + "/1_features/textured_sd_features.pt")[0].to(device)
            combined_feature1 = torch.cat([0.5*sd_feature1/torch.norm(sd_feature1, dim=-1, keepdim=True),0.5*dino_feature1/torch.norm(dino_feature1, dim=-1, keepdim=True)], dim=-1)
            combined_feature1 = combined_feature1/torch.norm(combined_feature1, dim=-1, keepdim=True)

            _, non_chirality_feature0, _ = model(combined_feature0.unsqueeze(0).float())
            _, non_chirality_feature1, _ = model(combined_feature1.unsqueeze(0).float())


            feature0 = torch.cat([mask_feature0, non_chirality_feature0[0]], dim=-1)
            feature1 = torch.cat([mask_feature1, non_chirality_feature1[0]], dim=-1)

        elif mode == "gt_chirality_mul":

            mask_feature0 = torch.tensor(np.loadtxt(full_path + "/0_chirality.txt"))[...,None].to(device=device)
            mask_feature1 = torch.tensor(np.loadtxt(full_path + "/1_chirality.txt"))[...,None].to(device=device)

            model = ChiralityDisentangler_SIGGRAPH(feature_size=3200+768, num_layers=2, model_type="mlp", normalization="beforeAndAfter", chirality_dim=1, force_orthogonal=True).to(device)
            model.load_state_dict(torch.load(model_path_siggraph, weights_only=True))

            dino_feature0 = torch.load(full_path + "/0_features/textured_dino_features.pt")[0].to(device)
            sd_feature0 = torch.load(full_path + "/0_features/textured_sd_features.pt")[0].to(device)
            combined_feature0 = torch.cat([0.5*sd_feature0/torch.norm(sd_feature0, dim=-1, keepdim=True),0.5*dino_feature0/torch.norm(dino_feature0, dim=-1, keepdim=True)], dim=-1)
            combined_feature0 = combined_feature0/torch.norm(combined_feature0, dim=-1, keepdim=True)

            dino_feature1 = torch.load(full_path + "/1_features/textured_dino_features.pt")[0].to(device)
            sd_feature1 = torch.load(full_path + "/1_features/textured_sd_features.pt")[0].to(device)
            combined_feature1 = torch.cat([0.5*sd_feature1/torch.norm(sd_feature1, dim=-1, keepdim=True),0.5*dino_feature1/torch.norm(dino_feature1, dim=-1, keepdim=True)], dim=-1)
            combined_feature1 = combined_feature1/torch.norm(combined_feature1, dim=-1, keepdim=True)

            _, non_chirality_feature0, _ = model(combined_feature0.unsqueeze(0).float())
            _, non_chirality_feature1, _ = model(combined_feature1.unsqueeze(0).float())


            feature0 = mask_feature0 * non_chirality_feature0[0]
            feature1 = mask_feature1 * non_chirality_feature1[0]

        cos_similarity = cosine_similarity(feature0, feature1).to(device=device)
        #cos_similarity = torch.randn(mesh0.vert.shape[0], mesh1.vert.shape[0]).to(device=device)
        print("similarity computation finished!")

        gt_corres = torch.tensor(np.load(full_path + "/corres_01.npy")).to(device=device)
        if len(gt_corres.shape) > 1:
            faces = torch.from_numpy(mesh1.face).to(device = device)
            gt_corres = faces[gt_corres[:, 0].long(), torch.argmax(gt_corres[:, 1:], -1)]

        verts0 = torch.tensor(mesh0.vert).to(device=device)
        verts1 = torch.tensor(mesh1.vert).to(device=device)

        result_dict = compute_acc(verts0, verts1, geodesic_1, cos_similarity, gt_corres)

        print("final results computed finished!")

        logs_step = {k: to_numpy(v) for k, v in result_dict.items()}
        tracks.append(logs_step)

        pairs.append(i)

        torch.save(cos_similarity, "/lustre/scratch/data/s95wwang_hpc-chirality_image/matching_faust_corres_prediction/" + mode  + str(i) + ".pt")

    df = pd.DataFrame(tracks)
    df.insert(0, "pair", pairs)
    df.to_csv("shape_matching_results_faust_" + mode + ".csv", index=False)