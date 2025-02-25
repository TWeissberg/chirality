import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions
from pymanopt.function import pytorch
import pymanopt

import scipy

from sklearn.utils.extmath import row_norms
from sklearn.cluster import kmeans_plusplus

import faiss

import os
from PIL import Image
import random

from torchvision import transforms as tfs

from dino import init_dino, get_dino_features
from diff3f import VERTEX_GPU_LIMIT, arange_pixels
from transformers import CLIPProcessor, CLIPVisionModel

from tqdm import tqdm

import pyvista as pv
from dataloaders.mesh_container import MeshContainer
import torchvision

import tempfile
import argparse
import wandb
from pathlib import Path
import glob
import time
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.structures import join_meshes_as_batch
import yaml
import pickle
import re

from prompt import get_prompt
import sys

torch.manual_seed(42)
np.random.seed(42)

device = "cuda"

# with open("./config.yaml") as file:
#     config = yaml.load(file, Loader=yaml.FullLoader)

os.environ["WANDB__SERVICE_WAIT"] = "300"

# config = [{ # normal neural network
#     "feature_type":"textured_sd_features",
#     "lambda_fifty":0.02979667670460797,
#     "lambda_inv":4.838802673866021,
#     "lambda_l1":201.46878408466037,
#     "lambda_orth":5.598372586811498,
#     "new_feature_dim":1,
#     "num_layers":2,
#     "invertible": False,
#     }, { # invertible neural networks
#     "feature_type":"textured_sd_features",
#     "invertible":True,
#     "lambda_fifty":0.003439786135804857,
#     "lambda_inv":0.00771276821093802,
#     "lambda_l1":377.3806589806351,
#     "lambda_orth":1.5107303299555348,
#     "new_feature_dim":1,
#     "num_layers":1,
#     }, { # lin
#     "feature_type":"textured_sd_features",
#     "invertible":True,
#     "lambda_fifty":0.006817046629229106,
#     "lambda_inv":0.0592494364990442,
#     "lambda_l1":973.1021708101838,
#     "lambda_orth":0.030646742112685324,
#     "new_feature_dim":1,
#     "num_layers":0,
# }][int(sys.argv[1])]

# config={
#     "feature_type":("textured_sd_features", "textured_dino_features"),
#     "invertible":True,
#     "lambda_fifty":0.006817046629229106,
#     "lambda_inv":0.0592494364990442,
#     "lambda_l1":973.1021708101838,
#     "lambda_orth":0.030646742112685324,
#     "new_feature_dim":1,
#     "num_layers":0,
#     "save_model": False,
# }

config = {
    "feature_type":("textured_sd_features", "textured_dino_features"),
    "invertible": False,
    "lambda_dis": 9500**0.5,
    "lambda_fifty":0.013666331402393064,
    "lambda_inv":2.3085700200712207 * (9500**0.5),
    "lambda_l1":276.92404142567483,
    "lambda_orth":0.0,#730.8342116110509,
    "new_feature_dim":1,
    "num_layers":2,
    "save_model":True,
    "check_illegal": False,
    }

# wandb.init(
# # set the wandb project where this run will be logged
# project="symmetry_legal",
# entity="tweissberg",
# # track hyperparameters and run metadata
# config=config
# )
print(config)

data_path = Path(os.path.expanduser(f"/lustre/scratch/data/tweissbe_hpc-becos/all/"))

def plot_and_save_html(V, F, assignment, name, keep = False):
	plotter = pv.Plotter(shape=(1, 1), off_screen=True)
	F = np.hstack([np.full((F.shape[0], 1), 3), F]).flatten()
	mesh = pv.PolyData(V, F)

	colors = plt.get_cmap("viridis") #create_colormap(assignment) # color map blue vs red per point (colors should have same shape as V) 

	plotter.add_mesh(mesh, scalars=colors(assignment), lighting=True, metallic=False, smooth_shading=True)

	plotter.camera_position = 'xy'
	plotter.link_views()
	plotter.render()
	# path_html = "./plot.html"
	if keep:
		path = name + ".html"
		plotter.export_html(path)
		plotter.close()
	else:
		with tempfile.NamedTemporaryFile(suffix=".html") as temp:
			path = str(temp.name)
			plotter.export_html(path)
			plotter.close()
			wandb.log({name: wandb.Html(path)})

def check_illegal_shape(shape_path):
    info_path = shape_path[:-4] + "_info.pkl"
    with open(info_path, "rb") as f:
        info = pickle.load(f)
        name = info["name"]
        dataset = info["org_dataset"]
    if dataset == "tosca":
        if name in ["david11", "michael5", "victoria12", "david0", "david6"]:
            print("Removed", name, dataset)
            return True
    elif dataset == "faust":
        if name in ["tr_reg_094", "test_reg_039", "tr_reg_068", "tr_reg_080", "tr_reg_090"]:
            print("Removed", name, dataset)
            return True
    elif dataset == "scape":
        if name in ["mesh012", "mesh014", "mesh030", "mesh033", "mesh067", "mesh069", "mesh071"]:
            print("Removed", name, dataset)
            return True
    return False

class BecosSymmetryDataset(Dataset):
    def __init__(self, data_dir, feature_type, num = None, category = "all", check_illegal = False):
        self.data_dir = data_dir
        pairs = glob.glob(str(data_dir / "*"))
        self.feature_type = feature_type
        shape_files = []
        feature_files = []
        chirality_files = []
        print(f"Searching {len(pairs)} pairs")
        for path in tqdm(pairs, "Finding shape and feature files 0"):
            shape_file0 = glob.glob(str(Path(path) / "0_*.off"))
            prompt0 = get_prompt(shape_file0[0])
            if (category == "human" and prompt0 != "human") or (category == "animal" and prompt0 == "human"):
                continue
            if check_illegal and check_illegal_shape(shape_file0[0]):
                continue
            feature_files0 = [tuple([glob.glob(str(Path(path) / f"0_features/{ft}.pt"))[0] for ft in feature_type])]
            chirality_file0 = glob.glob(str(Path(path) / f"0_chirality.txt"))
            if len(shape_file0) == 1 and len(chirality_file0) == 1:
                assert len(feature_files0[0]) == len(config["feature_type"])
                shape_files += shape_file0
                feature_files += feature_files0
                chirality_files += chirality_file0

        for path in tqdm(pairs, "Finding shape and feature files 1"):
            shape_file1 = glob.glob(str(Path(path) / "1_*.off"))
            prompt1 = get_prompt(shape_file1[0])
            if (category == "human" and prompt1 != "human") or (category == "animal" and prompt1 == "human"):
                continue
            if check_illegal and check_illegal_shape(shape_file1[0]):
                continue
            feature_files1 = [tuple([glob.glob(str(Path(path) / f"1_features/{ft}.pt"))[0] for ft in feature_type])]
            chirality_file1 = glob.glob(str(Path(path) / f"1_chirality.txt"))
            if len(shape_file1) == 1 and len(chirality_file1) == 1:
                assert len(feature_files1[0]) == len(wandb.config.feature_type)
                shape_files += shape_file1
                feature_files += feature_files1
                chirality_files += chirality_file1

        if category == "all" and not check_illegal:
            assert len(shape_files) == 2 * len(pairs)
        self.shape_files = shape_files[:num] if num is not None else shape_files
        self.feature_files = feature_files[:num] if num is not None else feature_files
        self.chirality_files = chirality_files[:num] if num is not None else chirality_files

    def __len__(self):
        assert len(self.shape_files) == len(self.feature_files)
        return len(self.shape_files)

    def __getitem__(self, idx):
        #start = time.time()
        shape = MeshContainer().load_from_file(self.shape_files[idx])
        features = torch.cat([torch.nn.functional.normalize(torch.load(x), 2) / len(self.feature_files[idx]) for x in self.feature_files[idx]], 2)
        chirality_info = torch.from_numpy(np.loadtxt(self.chirality_files[idx])).bool()
        #print(time.time() - start)
        return shape, features, chirality_info
    
        
def collate_fn(batch):
    return batch[0]


feature_size_dict = {
    "textured_dino_features": 768,
    "textured_clip_features": 1024,
    "textured_sd_features": 3200,
    "untextured_dino_features": 768,
    "untextured_clip_features": 1024,
    "untextured_sd_features": 3200,
}
feature_size = sum(feature_size_dict[ft] for ft in config["feature_type"])

print(feature_size)




A = torch.nn.Parameter((torch.eye(feature_size) + 0.1 * torch.rand(feature_size, feature_size)).to(device = device))
if config["num_layers"] == 0:
    forward_model = torch.nn.Identity()
    backward_model = torch.nn.Identity()
    optim = torch.optim.Adam([A], lr = 0.001)
elif config["invertible"]:
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm
    from torch import nn

    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, feature_size), nn.ReLU(),
                            nn.Linear(feature_size,  c_out))
    
    forward_model = Ff.SequenceINN(feature_size)
    for k in range(config["num_layers"]):
        forward_model.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)    #?
    optim = torch.optim.Adam(list(forward_model.parameters()) + [A], lr = 0.001)
    backward_model = None
    forward_model = forward_model.to(device = device)
else:
    forward_model = torchvision.ops.MLP(feature_size, [feature_size] * config["num_layers"]).to(device = device)
    backward_model = torchvision.ops.MLP(feature_size, [feature_size] * config["num_layers"]).to(device = device)
    optim = torch.optim.Adam(list(forward_model.parameters()) + list(backward_model.parameters()) + [A], lr = 0.001)

A = torch.load(os.path.join("best_legal", "A.h5"))
forward_dict = torch.load(os.path.join("best_legal", "forward.h5"))
forward_model.load_state_dict(forward_dict)
# if backward_model is not None:
#     torch.load(backward_model.state_dict(), os.path.join("best_legal", "backward.h5"))

def visualize(forward_model, backward_model, A, loader, num = 4, keep = True, harsh = False):
    plotter = pv.Plotter(shape=(1, num), off_screen=True)
    for i, dat in enumerate(loader):
        if i >= num:
            break

        mesh, both_features, chirality = dat
        both_features = both_features.to(device = device).float()
        both_features = torch.nn.functional.normalize(both_features, dim = -1)

        if not config["invertible"] or config["num_layers"] == 0:
            forward_features = forward_model(both_features)
        else:
            forward_features = torch.stack([forward_model(both_features[0])[0], forward_model(both_features[1])[0]])

        transformed_features = forward_features @ A
        transformed_features = torch.nn.functional.normalize(transformed_features, dim = -1)

        chirality_feature, _ = transformed_features[0, :].split([config["new_feature_dim"], transformed_features.shape[2] - config["new_feature_dim"]], -1)

        chirality_feature = chirality_feature[:, 0].detach().cpu().numpy()
        if harsh:
            if config["new_feature_dim"] == 1:
                chirality_feature = chirality_feature > 0
            else:
                chirality_feature = chirality_feature > 0.5

        V, F = mesh.vert, mesh.face
        F = np.hstack([np.full((F.shape[0], 1), 3), F]).flatten()
        mesh = pv.PolyData(V, F)

        colors = plt.get_cmap("viridis")

        plotter.subplot(0, i)
        plotter.add_mesh(mesh, scalars=colors(chirality_feature))#, lighting=True, metallic=False, smooth_shading=True)

    plotter.camera_position = 'xy'
    #plotter.link_views()
    plotter.render()
    # path_html = "./plot.html"
    if keep:
        path = "test.html"
        plotter.export_html(path)
        plotter.close()
    else:
        with tempfile.NamedTemporaryFile(suffix=".html") as temp:
            path = str(temp.name)
            plotter.export_html(path)
            plotter.close()
            wandb.log({"assignment_harsh" if harsh else "assignment": wandb.Html(path)})

from generate_features import load_model, calculate_pixel_to_vertices, features_from_images, get_dino_features, get_sd_features

dino_model = init_dino("cuda")
dino_feature_dim = 768
sd_model = load_model(diffusion_ver='v1-5', image_size=512, num_timesteps=50, block_indices=[2,5,8,11])
sd_feature_dim = 3200

torch.set_grad_enabled(False)
from utils import convert_mesh_container_to_torch_mesh
shape_file = data_path / "val" / "76" / "0_david10.off"
mesh = MeshContainer().load_from_file(shape_file)
mesh = convert_mesh_container_to_torch_mesh(mesh, device = "cuda", is_tosca = False)
feature_files = [data_path / "val" / "76" / "0_features" / f"{x}.pt" for x in config["feature_type"]]
features = torch.cat([torch.nn.functional.normalize(torch.load(x), 2) / len(feature_files) for x in feature_files], 2)
chirality_path = data_path / "val" / "76" / "0_chirality.txt"
chirality_info = torch.from_numpy(np.loadtxt(chirality_path)).bool()

images_path = data_path / "val" / "76" / "0_images" 
textured_images = torch.load(images_path / "textured.pt")
cameras = torch.load(images_path / "cameras.pt")
depth = torch.load(images_path / "depth.pt")

queried_indices = calculate_pixel_to_vertices(mesh, cameras, depth, bq = True)

H, W = 512, 512

hflip = tfs.functional.hflip

def hflip_indices(x):
    # Input : torch.Size([9, 262144, 100])
    n = x.shape[0]
    return hflip(x.permute(0, 2, 1).reshape(n, -1, H, W)).reshape(n, -1, H*W).permute(0, 2, 1)

grid = arange_pixels((H, W), invert_y_axis=False)[0].to("cuda").reshape(1, H, W, 2).half()

from torch.nn.functional import normalize
img_features = []
for img in tqdm(textured_images):
    dino_feature = get_dino_features("cuda", dino_model, img, grid.to(device = "cuda"))[0].T
    sd_feature = get_sd_features("cuda", sd_model, img, grid.to(device = "cuda"))[0].T
    img_features.append(torch.cat([0.5 * normalize(sd_feature), 0.5 * normalize(dino_feature)], dim = 1).cpu().detach())

img_features = torch.stack(img_features)


shape_features, num_features = features_from_images(mesh, textured_images, queried_indices, get_dino_features, dino_model, dino_feature_dim)
shape_features_flip, num_features_flip = features_from_images(mesh, [hflip(x) for x in textured_images], hflip_indices(queried_indices), get_dino_features, dino_model, dino_feature_dim)
textured_dino_features = torch.stack([shape_features, shape_features_flip])
assert (num_features == num_features_flip).all()